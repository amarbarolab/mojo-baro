from std.memory import ArcPointer
from grammar.automaton import Automaton, Bitset, MatcherState, Frame, resolve_epsilon, step_byte, step_byte_tracked
from grammar.trie import TokenTrie, TrieNode
from grammar.vocab import Vocab


@fieldwise_init
struct Snap(Copyable, Movable):
    var rule_id: Int32
    var state_id: Int32
    var call_stack: List[Frame]
    var terminated: Bool


struct Matcher(Copyable, Movable):
    var automaton: Automaton
    var trie: ArcPointer[TokenTrie]
    var vocab: ArcPointer[Vocab]
    var ms: MatcherState
    var snaps: List[Snap]

    def __init__(out self, var automaton: Automaton, trie: ArcPointer[TokenTrie], vocab: ArcPointer[Vocab]):
        self.automaton = automaton^
        self.trie = trie
        self.vocab = vocab
        self.ms = MatcherState(self.automaton.start_rule)
        resolve_epsilon(self.automaton, self.ms)
        self.snaps = []

    def is_terminated(self) -> Bool:
        return self.ms.terminated

    def snapshot(mut self) -> Int:
        self.snaps.append(Snap(self.ms.cur_rule, self.ms.cur_state, self.ms.call_stack.copy(), self.ms.terminated))
        return len(self.snaps) - 1

    def rollback(mut self, to: Int):
        ref s = self.snaps[to]
        self.ms.cur_rule = s.rule_id
        self.ms.cur_state = s.state_id
        self.ms.call_stack = s.call_stack.copy()
        self.ms.terminated = s.terminated

    def accept(mut self, token_id: Int) -> Bool:
        if self.vocab[].is_special[token_id]:
            return False
        var saved_rule = self.ms.cur_rule
        var saved_state = self.ms.cur_state
        var saved_stack = self.ms.call_stack.copy()
        var saved_term = self.ms.terminated
        ref bs = self.vocab[].token_bytes[token_id]
        var bytes_ok = True
        for i in range(len(bs)):
            if not step_byte(self.automaton, self.ms, bs[i]):
                bytes_ok = False
                break
            resolve_epsilon(self.automaton, self.ms)
        if not bytes_ok:
            self.ms.cur_rule = saved_rule
            self.ms.cur_state = saved_state
            self.ms.call_stack = saved_stack^
            self.ms.terminated = saved_term
            return False
        return True

    def fill_mask(mut self, mut mask: Bitset):
        mask.clear_all()
        var scratch: List[Frame] = []
        _walk(self.automaton, self.trie[], self.ms, 0, mask, scratch)


def _walk(automaton: Automaton, trie: TokenTrie, mut ms: MatcherState, node_idx: Int, mut mask: Bitset, mut scratch: List[Frame]):
    ref node = trie.nodes[node_idx]
    if node.token_id >= 0:
        mask.set_bit(Int(node.token_id))
    for i in range(len(node.child_byte)):
        var b = node.child_byte[i]
        var saved_rule = ms.cur_rule
        var saved_state = ms.cur_state
        var saved_term = ms.terminated
        var scratch_start = len(scratch)
        if step_byte_tracked(automaton, ms, b, scratch):
            var stack_len_after_step = len(ms.call_stack)
            resolve_epsilon(automaton, ms)
            _walk(automaton, trie, ms, Int(node.child_idx[i]), mask, scratch)
            while len(ms.call_stack) > stack_len_after_step:
                _ = ms.call_stack.pop()
            var k = len(scratch)
            for j in range(k - 1, scratch_start - 1, -1):
                ms.call_stack.append(scratch[j])
            while len(scratch) > scratch_start:
                _ = scratch.pop()
        ms.cur_rule = saved_rule
        ms.cur_state = saved_state
        ms.terminated = saved_term
