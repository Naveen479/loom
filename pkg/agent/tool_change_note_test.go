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
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/teradata-labs/loom/pkg/llm"
	llmtypes "github.com/teradata-labs/loom/pkg/llm/types"
	"github.com/teradata-labs/loom/pkg/shuttle"
)

func TestToolDeltaNote(t *testing.T) {
	assert.Equal(t, "", toolDeltaNote([]string{"a", "b"}, []string{"b", "a"}), "same set, other order")
	assert.Equal(t, "", toolDeltaNote(nil, nil))
	assert.Equal(t, "Tools added: a, c.", toolDeltaNote([]string{"b"}, []string{"c", "b", "a"}), "add only, sorted")
	assert.Equal(t, "Tools removed: b, c.", toolDeltaNote([]string{"a", "c", "b"}, []string{"a"}), "remove only, sorted")
	assert.Equal(t, "Tools added: x. Tools removed: a.", toolDeltaNote([]string{"a", "b"}, []string{"b", "x"}), "both")
}

// revealTool registers a session-scoped tool while the first tool of the turn
// executes, the way a progressive-disclosure activation does.
type revealTool struct {
	namedStubTool
	ag        *Agent
	sessionID string
	once      sync.Once
}

func (r *revealTool) Execute(context.Context, map[string]interface{}) (*shuttle.Result, error) {
	r.once.Do(func() {
		late := namedStubTool{name: "late_tool"}
		r.ag.tools.Register(late)
		r.ag.registerSessionTool(r.sessionID, late.name)
	})
	return &shuttle.Result{Success: true, Data: "ok"}, nil
}

// The note rides exactly the provider call that first sees a changed tool set,
// is never stored in session history, and is not repeated on a later call with
// no further change.
func TestAgent_ToolChangeNoteOnlyOnChangedCall(t *testing.T) {
	const sid = "tool_change_note"
	llm := &recordingToolLLM{mockToolCallingLLM: &mockToolCallingLLM{
		responses: []mockLLMResponse{
			{toolCalls: []llmtypes.ToolCall{{ID: "c1", Name: "reveal", Input: map[string]interface{}{}}}},
			{toolCalls: []llmtypes.ToolCall{{ID: "c2", Name: "reveal", Input: map[string]interface{}{}}}},
			{content: "done"},
		},
	}}
	ag := NewAgent(&mockBackend{}, llm, WithConfig(&Config{MaxTurns: 10, MaxToolExecutions: 10, MaxIterations: 10}))
	ag.RegisterTool(&revealTool{namedStubTool: namedStubTool{name: "reveal"}, ag: ag, sessionID: sid})

	_, err := ag.Chat(context.Background(), sid, "use the tool")
	require.NoError(t, err)

	llm.recMu.Lock()
	defer llm.recMu.Unlock()
	require.Len(t, llm.seen, 3)
	lastOf := func(msgs []llmtypes.Message) string { return msgs[len(msgs)-1].Content }

	assert.NotContains(t, lastOf(llm.seen[0]), "Tools added", "first send of a turn compares with nothing")
	assert.Contains(t, lastOf(llm.seen[1]), "<system-reminder>\nTools added: late_tool.\n</system-reminder>")
	assert.NotContains(t, lastOf(llm.seen[2]), "Tools added", "no change, no note")
	assert.NotContains(t, lastOf(llm.seen[2]), "Tools removed", "no change, no note")

	session, ok := ag.GetSession(sid)
	require.True(t, ok)
	for _, m := range session.GetMessages() {
		assert.NotContains(t, m.Content, "Tools added", "the note must never be persisted")
	}
}

// resendLLM reveals late2 on the refused send (call 2) and refuses it with a
// context-too-long error, so the resend carries a tool set the first send did
// not. Call 3 answers.
type resendLLM struct {
	*mockToolCallingLLM
	ag   *Agent
	sid  string
	mu   sync.Mutex
	seen [][]llmtypes.Message
}

func (r *resendLLM) Chat(_ context.Context, messages []llmtypes.Message, _ []shuttle.Tool) (*llmtypes.LLMResponse, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.seen = append(r.seen, append([]llmtypes.Message(nil), messages...))
	switch len(r.seen) {
	case 1:
		return &llmtypes.LLMResponse{ToolCalls: []llmtypes.ToolCall{{ID: "c1", Name: "reveal", Input: map[string]interface{}{}}}}, nil
	case 2:
		r.ag.tools.Register(namedStubTool{name: "late2"})
		r.ag.registerSessionTool(r.sid, "late2")
		return nil, fmt.Errorf("stream error: 400 prompt is too long: %w", llm.ErrContextTooLong)
	}
	return &llmtypes.LLMResponse{Content: "done"}, nil
}

// The resend is a second provider call of the same iteration. Its note must
// describe the tools that call actually sends, and the refused send must not
// become the baseline.
func TestAgent_ToolChangeNoteResendUsesSentTools(t *testing.T) {
	const sid = "tool_change_resend"
	llm := &resendLLM{mockToolCallingLLM: &mockToolCallingLLM{}, sid: sid}
	ag := NewAgent(&mockBackend{}, llm, WithConfig(&Config{MaxTurns: 10, MaxToolExecutions: 10, MaxIterations: 10}))
	llm.ag = ag
	ag.RegisterTool(&revealTool{namedStubTool: namedStubTool{name: "reveal"}, ag: ag, sessionID: sid})

	_, err := ag.Chat(context.Background(), sid, "use the tool")
	require.NoError(t, err)

	llm.mu.Lock()
	defer llm.mu.Unlock()
	require.Len(t, llm.seen, 3)
	lastOf := func(msgs []llmtypes.Message) string { return msgs[len(msgs)-1].Content }
	assert.Contains(t, lastOf(llm.seen[1]), "Tools added: late_tool.", "the refused send carries its own note")
	assert.Contains(t, lastOf(llm.seen[2]), "Tools added: late2, late_tool.", "the resend diffs against the last successful send")
	assert.NotContains(t, lastOf(llm.seen[2]), "Tools added: late_tool.\n", "no stale note from the refused send")
}
