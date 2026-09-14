// Package max implements a small, evidence-oriented Arabic understanding core.
// It deliberately keeps linguistic analysis, semantic interpretation, learned
// evidence, and durable memory as separate owners of state.
package max

import (
	"encoding/json"
	"errors"
	"sort"
	"strings"
	"sync"
	"unicode"
)

// Token is morphology's output. It describes a surface form only; it does not
// own semantic meaning.
type Token struct {
	Surface, Stem             string
	Definite, Future, Negated bool
}

// Parse is syntax's output. Roles are structural hypotheses, never facts.
type Parse struct {
	Tokens                     []Token
	Subject, Predicate, Object int
	Confidence                 float64
}

// Interpretation is the single semantic view of an utterance.
type Interpretation struct {
	Predicate  string   `json:"predicate"`
	Agent      string   `json:"agent,omitempty"`
	Patient    string   `json:"patient,omitempty"`
	Tense      string   `json:"tense"`
	Negated    bool     `json:"negated"`
	Confidence float64  `json:"confidence"`
	Unknown    []string `json:"unknown,omitempty"`
}

// Evidence records what was observed. Claims are intentionally not inferred
// facts: callers must promote them through their own verification policy.
type Evidence struct {
	Observation, Meaning, Source string
	Count                        int
}

// Episode preserves a supervised training event separately from lexical
// evidence. Its expected interpretation is a label supplied by a caller, not a
// fact asserted by MAX.
type Episode struct {
	Input    string         `json:"input"`
	Expected Interpretation `json:"expected"`
}

// Memory has distinct durable stores for observations, hypotheses and verified
// facts. This prevents a learned guess from silently becoming a fact.
type Memory struct {
	Observations []Evidence `json:"observations"`
	Hypotheses   []Evidence `json:"hypotheses"`
	Facts        []Evidence `json:"facts"`
	Episodes     []Episode  `json:"episodes"`
}

// Engine owns the pipeline and learned lexical evidence. It is safe to persist
// through Snapshot/Restore; no parser state is serialized as semantic truth.
type Engine struct {
	mu      sync.RWMutex
	lexicon map[string]Evidence
	memory  Memory
}

func New() *Engine { return &Engine{lexicon: map[string]Evidence{}} }

func tokenize(text string) []Token {
	words := strings.FieldsFunc(strings.TrimSpace(text), func(r rune) bool { return unicode.IsPunct(r) || unicode.IsSpace(r) })
	out := make([]Token, 0, len(words))
	for _, w := range words {
		t := Token{Surface: w, Stem: w}
		if strings.HasPrefix(t.Stem, "لا") && len([]rune(t.Stem)) > 2 {
			t.Negated, t.Stem = true, strings.TrimPrefix(t.Stem, "لا")
		}
		if t.Stem == "لا" || t.Stem == "لن" || t.Stem == "لم" || t.Stem == "ليس" {
			t.Negated = true
		}
		// The future clitic only attaches to a verb. Checking the first letter of
		// its remainder avoids corrupting names such as سامي.
		if strings.HasPrefix(t.Stem, "س") && len([]rune(t.Stem)) > 3 &&
			(strings.HasPrefix(strings.TrimPrefix(t.Stem, "س"), "ي") || strings.HasPrefix(strings.TrimPrefix(t.Stem, "س"), "ت") || strings.HasPrefix(strings.TrimPrefix(t.Stem, "س"), "أ") || strings.HasPrefix(strings.TrimPrefix(t.Stem, "س"), "ن")) {
			t.Future, t.Stem = true, strings.TrimPrefix(t.Stem, "س")
		}
		if strings.HasPrefix(t.Stem, "ال") && len([]rune(t.Stem)) > 3 {
			t.Definite, t.Stem = true, strings.TrimPrefix(t.Stem, "ال")
		}
		out = append(out, t)
	}
	return out
}

// ParseArabic uses order and morphological markers rather than sentence
// memorisation: verb-initial clauses are VSO; otherwise they are SVO.
func ParseArabic(text string) Parse {
	tokens := tokenize(text)
	p := Parse{Tokens: tokens, Subject: -1, Predicate: -1, Object: -1, Confidence: .35}
	content := make([]int, 0, len(tokens))
	for i, t := range tokens {
		if t.Stem != "" && t.Surface != "لا" && t.Surface != "لن" && t.Surface != "لم" && t.Surface != "ليس" {
			content = append(content, i)
		}
	}
	if len(content) == 0 {
		return p
	}
	verb := func(t Token) bool {
		return strings.HasPrefix(t.Stem, "ي") || strings.HasPrefix(t.Stem, "ت") || strings.HasPrefix(t.Stem, "أ") || strings.HasPrefix(t.Stem, "ن") || strings.HasSuffix(t.Stem, "ت")
	}
	if verb(tokens[content[0]]) {
		p.Predicate = content[0]
		if len(content) > 1 {
			p.Subject = content[1]
		}
		if len(content) > 2 {
			p.Object = content[2]
		}
	} else {
		p.Subject = content[0]
		if len(content) > 1 {
			p.Predicate = content[1]
		}
		if len(content) > 2 {
			p.Object = content[2]
		}
	}
	p.Confidence = .75
	return p
}

// Learn stores labelled lexical observations. A label may apply to a previously
// unseen sentence because interpretation consumes the learned stem, not text.
func (e *Engine) Learn(word, meaning string) error {
	return e.learn(word, meaning, "direct")
}

func (e *Engine) learn(word, meaning, source string) error {
	stem := tokenize(word)
	if len(stem) != 1 || stem[0].Stem == "" || strings.TrimSpace(meaning) == "" {
		return errors.New("learning requires one non-empty word and meaning")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	v := e.lexicon[stem[0].Stem]
	if v.Count > 0 && v.Meaning != meaning {
		// Conflicting labels are evidence about uncertainty, not permission to
		// overwrite established meaning or manufacture a fact.
		e.memory.Hypotheses = append(e.memory.Hypotheses, Evidence{Observation: stem[0].Stem, Meaning: meaning, Source: source, Count: 1})
		return errors.New("conflicting evidence retained as hypothesis")
	}
	v.Observation, v.Meaning, v.Source, v.Count = stem[0].Stem, meaning, source, v.Count+1
	e.lexicon[stem[0].Stem] = v
	e.memory.Observations = append(e.memory.Observations, v)
	return nil
}

// LearnExample is the supervised-training entry point. It induces three
// word-level semantic observations from a labelled clause, after first using
// the structural parser to align predicate, agent, and patient. It never stores
// a whole sentence in the lexicon, so later success on a new combination is
// necessarily compositional.
func (e *Engine) LearnExample(input string, expected Interpretation) error {
	p := ParseArabic(input)
	if p.Predicate < 0 || p.Subject < 0 || p.Object < 0 {
		return errors.New("training example requires predicate, agent, and patient")
	}
	labels := []struct {
		index int
		value string
	}{{p.Predicate, expected.Predicate}, {p.Subject, expected.Agent}, {p.Object, expected.Patient}}
	for _, label := range labels {
		if strings.TrimSpace(label.value) == "" {
			return errors.New("training example has an empty semantic role")
		}
	}
	// Validate conflicts before updating any evidence, making an example update
	// atomic at the semantic level.
	e.mu.RLock()
	for _, label := range labels {
		stem := p.Tokens[label.index].Stem
		if known := e.lexicon[stem]; known.Count > 0 && known.Meaning != label.value {
			e.mu.RUnlock()
			return errors.New("training example conflicts with existing evidence")
		}
	}
	e.mu.RUnlock()
	for _, label := range labels {
		if err := e.learn(p.Tokens[label.index].Surface, label.value, "supervised-example"); err != nil {
			return err
		}
	}
	e.mu.Lock()
	e.memory.Episodes = append(e.memory.Episodes, Episode{Input: input, Expected: expected})
	e.mu.Unlock()
	return nil
}

func (e *Engine) Understand(text string) Interpretation {
	p := ParseArabic(text)
	r := Interpretation{Tense: "present", Confidence: p.Confidence}
	if p.Predicate < 0 {
		r.Confidence = 0
		return r
	}
	for _, t := range p.Tokens {
		if t.Negated {
			r.Negated = true
		}
		if t.Future {
			r.Tense = "future"
		}
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	resolve := func(i int) string {
		if i < 0 {
			return ""
		}
		s := p.Tokens[i].Stem
		if v, ok := e.lexicon[s]; ok {
			return v.Meaning
		}
		r.Unknown = append(r.Unknown, s)
		return s
	}
	r.Predicate = resolve(p.Predicate)
	r.Agent = resolve(p.Subject)
	r.Patient = resolve(p.Object)
	// Confidence reflects the proportion of unresolved semantic roles. Unknown
	// vocabulary remains explicit rather than being replaced with a guess.
	roles := 1
	if p.Subject >= 0 {
		roles++
	}
	if p.Object >= 0 {
		roles++
	}
	r.Confidence *= 1 - .4*float64(len(r.Unknown))/float64(roles)
	return r
}

// Memory returns a deep copy, preserving Engine as the sole owner of mutable
// semantic and evidence state.
func (e *Engine) Memory() Memory {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return cloneMemory(e.memory)
}

func cloneMemory(m Memory) Memory {
	m.Observations = append([]Evidence(nil), m.Observations...)
	m.Hypotheses = append([]Evidence(nil), m.Hypotheses...)
	m.Facts = append([]Evidence(nil), m.Facts...)
	m.Episodes = append([]Episode(nil), m.Episodes...)
	return m
}

func (e *Engine) Snapshot() ([]byte, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return json.Marshal(e.memory)
}
func (e *Engine) Restore(data []byte) error {
	var m Memory
	if err := json.Unmarshal(data, &m); err != nil {
		return err
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	e.memory = cloneMemory(m)
	e.lexicon = map[string]Evidence{}
	for _, v := range m.Observations {
		old := e.lexicon[v.Observation]
		if v.Count > old.Count {
			e.lexicon[v.Observation] = v
		}
	}
	return nil
}
func (e *Engine) KnownWords() []string {
	e.mu.RLock()
	defer e.mu.RUnlock()
	out := make([]string, 0, len(e.lexicon))
	for k := range e.lexicon {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
