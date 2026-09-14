package max

import (
	"fmt"
	"testing"
)

func trained(t *testing.T) *Engine {
	t.Helper()
	e := New()
	for w, m := range map[string]string{"يقرأ": "read", "سامي": "sami", "كتاب": "book", "لورا": "lora"} {
		if err := e.Learn(w, m); err != nil {
			t.Fatal(err)
		}
	}
	return e
}

func TestKnownVSO(t *testing.T) {
	r := trained(t).Understand("يقرأ سامي الكتاب")
	if r.Predicate != "read" || r.Agent != "sami" || r.Patient != "book" {
		t.Fatalf("bad semantics: %+v", r)
	}
}
func TestUnseenCompositionGeneralizes(t *testing.T) {
	r := trained(t).Understand("يقرأ لورا كتاب")
	if r.Predicate != "read" || r.Agent != "lora" || r.Patient != "book" {
		t.Fatalf("did not generalize: %+v", r)
	}
}
func TestStructuralPermutationUsesSVO(t *testing.T) {
	r := trained(t).Understand("سامي يقرأ كتاب")
	if r.Predicate != "read" || r.Agent != "sami" || r.Patient != "book" {
		t.Fatalf("SVO structure was not interpreted: %+v", r)
	}
}
func TestNovelWordRetainsUncertainty(t *testing.T) {
	r := trained(t).Understand("يقرأ زاف كتاب")
	if len(r.Unknown) != 1 || r.Unknown[0] != "زاف" || r.Confidence >= .75 {
		t.Fatalf("unknown word fabricated knowledge: %+v", r)
	}
}
func TestNegationAndFutureAreCompositional(t *testing.T) {
	r := trained(t).Understand("لا سيقرأ سامي كتاب")
	if !r.Negated || r.Tense != "future" || r.Predicate != "read" {
		t.Fatalf("markers lost: %+v", r)
	}
}
func TestLearningImprovesUnseenSentence(t *testing.T) {
	e := New()
	before := e.Understand("يقرأ زاف")
	if err := e.Learn("يقرأ", "read"); err != nil {
		t.Fatal(err)
	}
	// This composition was never presented to Learn.
	after := e.Understand("يقرأ لورا")
	if after.Predicate != "read" || after.Confidence <= before.Confidence {
		t.Fatalf("no measurable improvement: before=%+v after=%+v", before, after)
	}
}
func TestRestartPreservesEvidenceNotFacts(t *testing.T) {
	e := trained(t)
	data, err := e.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	restored := New()
	if err := restored.Restore(data); err != nil {
		t.Fatal(err)
	}
	r := restored.Understand("يقرأ سامي كتاب")
	if r.Predicate != "read" || len(restored.Memory().Facts) != 0 {
		t.Fatalf("restart or truth boundary failed: %+v", restored.Memory())
	}
}

func TestConflictingLearningStaysHypothesis(t *testing.T) {
	e := New()
	if err := e.Learn("يقرأ", "read"); err != nil {
		t.Fatal(err)
	}
	if err := e.Learn("يقرأ", "write"); err == nil {
		t.Fatal("conflicting label was accepted as knowledge")
	}
	r := e.Understand("يقرأ")
	if r.Predicate != "read" || len(e.Memory().Hypotheses) != 1 || len(e.Memory().Facts) != 0 {
		t.Fatalf("evidence boundary failed: result=%+v memory=%+v", r, e.Memory())
	}
}

func TestSupervisedExampleLearnsRolesWithoutSentenceLookup(t *testing.T) {
	e := New()
	if err := e.LearnExample("يحلل نور بيانات", Interpretation{Predicate: "analyse", Agent: "noor", Patient: "data"}); err != nil {
		t.Fatal(err)
	}
	// The target sentence was not supplied during training; only its lexical
	// roles can transfer to this new VSO composition.
	if err := e.Learn("ريم", "reem"); err != nil {
		t.Fatal(err)
	}
	r := e.Understand("يحلل ريم بيانات")
	if r.Predicate != "analyse" || r.Agent != "reem" || r.Patient != "data" || len(e.Memory().Episodes) != 1 {
		t.Fatalf("supervised transfer failed: result=%+v memory=%+v", r, e.Memory())
	}
}

func TestConflictingExampleDoesNotPartiallyUpdate(t *testing.T) {
	e := New()
	if err := e.LearnExample("يحلل نور بيانات", Interpretation{Predicate: "analyse", Agent: "noor", Patient: "data"}); err != nil {
		t.Fatal(err)
	}
	if err := e.LearnExample("يحلل نور ملف", Interpretation{Predicate: "rewrite", Agent: "noor", Patient: "file"}); err == nil {
		t.Fatal("conflicting example accepted")
	}
	if got := len(e.Memory().Observations); got != 3 {
		t.Fatalf("partial update leaked into memory: %d observations", got)
	}
}

// TestHalfMillionArabicCompositions is a generated, independent evaluator. It
// trains only 250 lexical observations, then evaluates 500,000 unique Arabic
// clauses. No complete sentence is passed to Learn, so success requires role
// composition plus reuse of learned lexical evidence rather than memorisation.
func TestHalfMillionArabicCompositions(t *testing.T) {
	e := New()
	const verbs, agents, patients = 100, 100, 50
	for i := 0; i < verbs; i++ {
		if err := e.Learn(fmt.Sprintf("يعمل%d", i), fmt.Sprintf("action:%d", i)); err != nil {
			t.Fatal(err)
		}
	}
	for i := 0; i < agents; i++ {
		if err := e.Learn(fmt.Sprintf("رامي%d", i), fmt.Sprintf("agent:%d", i)); err != nil {
			t.Fatal(err)
		}
	}
	for i := 0; i < patients; i++ {
		if err := e.Learn(fmt.Sprintf("عنصر%d", i), fmt.Sprintf("patient:%d", i)); err != nil {
			t.Fatal(err)
		}
	}
	if got := len(e.Memory().Observations); got != verbs+agents+patients {
		t.Fatalf("unexpected training footprint: %d", got)
	}

	checked := 0
	for v := 0; v < verbs; v++ {
		for a := 0; a < agents; a++ {
			for p := 0; p < patients; p++ {
				var sentence string
				if checked%2 == 0 {
					sentence = fmt.Sprintf("يعمل%d رامي%d عنصر%d", v, a, p)
				} else {
					sentence = fmt.Sprintf("رامي%d يعمل%d عنصر%d", a, v, p)
				}
				r := e.Understand(sentence)
				if r.Predicate != fmt.Sprintf("action:%d", v) || r.Agent != fmt.Sprintf("agent:%d", a) || r.Patient != fmt.Sprintf("patient:%d", p) || len(r.Unknown) != 0 {
					t.Fatalf("failed unseen composition %q: %+v", sentence, r)
				}
				checked++
			}
		}
	}
	if checked != 500000 {
		t.Fatalf("evaluator coverage = %d, want 500000", checked)
	}
}
