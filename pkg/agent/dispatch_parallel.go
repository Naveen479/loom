// Copyright 2026 Teradata
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package agent

import (
	"fmt"
	"sync"

	"github.com/teradata-labs/loom/pkg/shuttle"
)

// maxParallelReads bounds how many tool calls of one group run at once.
const maxParallelReads = 4

// dispatchBatch dispatches one model response's tool calls.
//
// With Config.ParallelReadTools off (the default) it is the sequential loop:
// each call through dispatchOneCall, stopping at MaxToolExecutions.
//
// With it on, consecutive group-eligible calls (see parallelGroupLen) form a
// group: prepareCall for each member in call order, runCall for all members
// concurrently, then settleCall for each member in call order. Any other call
// is a barrier, dispatched alone through dispatchOneCall after the group
// before it has fully settled. Rows, execution records and dedup entries
// therefore land in call order whatever order the tools finish in. Progress
// events of a group fire as all tool-started, then all tool-completed; their
// percentage comes from the execution counter, so it differs slightly from a
// sequential run.
//
// A failing tool inside a group cannot trip its circuit breaker for the other
// members of the same group: they all passed the breaker before any ran, so up
// to maxParallelReads calls can reach a failing tool before the breaker opens.
// A safe tool that panics is turned into an error row only inside a group; a
// lone call still takes the sequential path, where the panic is not recovered.
//
// ResumeChat does not use this: its parked-batch completion stays sequential.
func (a *Agent) dispatchBatch(ctx Context, session *Session, calls []ToolCall, st *batchState) {
	if !a.config.ParallelReadTools {
		for i, toolCall := range calls {
			if *st.toolExecutionCount >= a.config.MaxToolExecutions {
				break
			}
			a.dispatchOneCall(ctx, session, toolCall, i, st)
		}
		return
	}

	for i := 0; i < len(calls); {
		if *st.toolExecutionCount >= a.config.MaxToolExecutions {
			return
		}
		if n := a.parallelGroupLen(calls[i:], st); n > 1 {
			i += a.dispatchGroup(ctx, session, calls[i:i+n], i, st)
			continue
		}
		a.dispatchOneCall(ctx, session, calls[i], i, st)
		i++
	}
}

// parallelGroupLen returns how many calls at the head of calls may run as one
// concurrent group. A call is eligible only if its tool resolves and declares
// shuttle.ConcurrencySafe, it is not contact_human, and its dedup key is not
// already cached this turn nor repeated earlier in the group (a duplicate must
// wait to replay the first call's result).
//
// An admission chain makes every call a barrier: its hooks are order-dependent
// across one batch (a render call records an approval a later gated call
// reads, and post-tool hooks run inside Execute), so the flag then degrades to
// sequential dispatch.
//
// The length is also capped by the remaining MaxToolExecutions and per-turn
// budgets, so prepareCall can never commit a turn_limit_exceeded row while an
// earlier group member is still unsettled.
func (a *Agent) parallelGroupLen(calls []ToolCall, st *batchState) int {
	if a.executor.HasAdmissionChain() {
		return 0
	}
	limit := min(maxParallelReads,
		a.config.MaxToolExecutions-*st.toolExecutionCount,
		st.maxPerTurn-st.turnToolCount,
		len(calls))
	seen := make(map[string]bool, limit)
	n := 0
	for ; n < limit; n++ {
		c := calls[n]
		key := c.Name + "|" + canonicalJSON(c.Input)
		if _, cached := st.turnDedup[key]; cached || seen[key] || !a.groupEligibleTool(c.Name) {
			break
		}
		seen[key] = true
	}
	return n
}

// groupEligibleTool resolves name the way the executor does (direct registry
// hit, else the canonical name) and reports whether the tool declares itself
// concurrency-safe. An unresolved tool is not eligible (fail closed).
func (a *Agent) groupEligibleTool(name string) bool {
	if name == "contact_human" {
		return false
	}
	tool, ok := a.tools.Get(name)
	if !ok {
		tool, ok = a.tools.Get(a.executor.CanonicalToolName(name))
	}
	return ok && shuttle.IsConcurrencySafe(tool)
}

// dispatchGroup runs one group and returns how many calls it consumed.
func (a *Agent) dispatchGroup(ctx Context, session *Session, calls []ToolCall, first int, st *batchState) int {
	consumed := len(calls)
	pcs := make([]*preparedCall, 0, len(calls))
	for j, c := range calls {
		pc, finished := a.prepareCall(ctx, session, c, first+j, st)
		// Unreachable by construction: parallelGroupLen's caps and dedup
		// checks mean no member can hit prepareCall's cap or dedup row. If it
		// ever fires, that row is already committed before the members
		// prepared above settle, so row order for this batch is not
		// guaranteed.
		if finished {
			consumed = j + 1
			break
		}
		pcs = append(pcs, pc)
	}

	results := make([]*shuttle.Result, len(pcs))
	errs := make([]error, len(pcs))
	var wg sync.WaitGroup
	for j, pc := range pcs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			// A panicking tool (some wrap panic-based libraries) becomes this
			// member's error; the other members run and settle as usual.
			defer func() {
				if r := recover(); r != nil {
					results[j], errs[j] = nil, fmt.Errorf("tool %s panicked: %v", pc.call.Name, r)
				}
			}()
			// recover() does not see runtime.Goexit; without this preset a
			// member that exits that way settles as (nil, nil) and settleCall
			// nil-derefs. A normal return overwrites it.
			errs[j] = fmt.Errorf("tool %s did not return", pc.call.Name)
			results[j], errs[j] = a.runCall(ctx, pc, session.ID)
		}()
	}
	wg.Wait()

	for j, pc := range pcs {
		a.settleCall(ctx, session, pc, st, results[j], errs[j])
	}
	return consumed
}
