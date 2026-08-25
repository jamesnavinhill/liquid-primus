Writing prompts for Transformer Lab Primus
Your prompt is the starting point of a research project: a direction you want explored, rather than a spec for a finished model. Aim to point Primus at something worth investigating.

Vague is fine. Primus asks follow-up questions and sharpens the project with you before the work starts, so a one-sentence idea is enough to begin. This guide is about picking a direction that tends to lead somewhere.

You don't need much
Two prompts that started projects:

"Improve inference time on (insert mode name) and get it to fit on a 24GB GPU without sacrificing coding performance."

"I think LLMs might hallucinate less if they were aware of their own token probabilities. Help me figure out an experiment around that."

Neither names a method, a metric, or a dataset, and both ended in a publishable paper. Write the version you can write now. If you don't know a detail, leave it out or say you're unsure.

What's worth including (if you have it)
None of these are required. They just give the project a sharper start when you happen to know them:

What you're curious about, and why it interests you. The question matters more than the technique.
Anything to compare against: a benchmark, a dataset, a paper's claim you want to check, an existing method you think you can beat.
Hard limits: compute you have, licenses, anything that's explicitly not part of this.
Your priority when two goals conflict. For example, "I care about calibration as much as raw accuracy."
If you don't have these, don't invent them. Say "not sure" and Primus will work them out with you.

Data: bring it, or let Primus find it
Don't get stuck because you don't have a dataset lined up. Primus searches out public data that fits your question, and generates data synthetically when nothing suitable exists. "I'm not sure what data to use" is a workable answer.

If you do have something specific in mind, hand it over: paste a link to a dataset, a Hugging Face model, an eval suite, or a git repo, and Primus will pull from there instead of going looking. The same goes for a particular benchmark you want to be measured against.

Kinds of questions that tend to work
Each of these shapes has produced a finished project. Use them to find a direction.

Pattern	What it looks like	Example
Does X cause Y?	A mechanism you suspect drives some behavior.	"Does the RL clipping mechanism control how diverse the generated outputs are?"
Is that claim true?	A published or vendor result you want to check independently.	"A vendor says their model wins on this metric. Does the win survive a better metric?"
Open question, pick a side	Something the field hasn't cleanly settled.	"Does post-training move a model's representations toward or away from how brains process language?"
Beat the incumbent	Push past a strong baseline under fair, matched conditions.	"Beat directed evolution at proposing high-fitness protein variants at the same query budget."
Build the yardstick	No trusted metric/benchmark exists yet, so make and validate one.	"There's no trusted automatic score for 3D-mesh quality, so build one and show it tracks human judgment."
Fix a known failure	Apply a method to a documented weakness.	"Use preference optimization to fix the word-skip/repeat failures TTS models are known for."
Port it / replatform	Re-implement a known model on a different architecture or stack, and prove it still works.	"Replatform this model onto architecture X and verify it holds up across the key benchmarks."
Optimize one dimension	Push a single efficiency axis with the quality bar held fixed.	"Find the best quantization of model X, judged on X and Y, without losing accuracy on Z."
I have a hunch — help me shape it	A theory you want turned into an experiment.	"Maybe injecting a self-reflection token reduces hallucination, so let's design something to test it."
Four of them come with a yardstick built in: checking a claim, beating an incumbent, porting a model, and optimizing one dimension each give you something concrete to be right or wrong against, whether that's a benchmark to match, a baseline to beat, or a quality bar to hold. Start there if you're new to a subfield.

Aim narrow, not for the moon
Prompts go wrong by being too grand more often than by being too vague. An enormous, open-ended question ("invent an architecture better than transformers", "solve hallucination") leaves the project no edge to push on and no way to tell whether it succeeded.

The fix is almost always to name a specific weakness and investigate improving it without breaking everything else. Instead of "beat transformers," try: "transformers struggle with long-context retrieval, so explore changes that improve it while holding language modeling and short-context quality fixed." That version has a target to move and a guardrail against regressing the rest.

Make sure the experiment can actually run
Primus can take on ambitious work, full fine-tunes of large open models included. Ask it to fine-tune something like GLM 5.2 and it will figure out how. But Primus will ask you what your budget is, and it can only do experiments that fit within the hardware and budget you have access to.

So when your idea implies heavy training, say what you want spent, or how big you want to go: "keep this to a single GPU," or "I'm not sure what this needs, size it for me." Primus will scope the experiment to fit the budget, or tell you up front what a larger version would cost, and it stops to ask rather than quietly spending past the cap you approved.

A few things to avoid
Asking for "a model" with no question behind it. "Make a better TTS model" gives the project nothing to aim at; name the failure you want fixed instead.
Locking in choices you're unsure about. If you name a base model or metric, it'll be taken as fixed. If you're not sure, float it as a suggestion ("maybe XTTS or Fish-Speech, whichever fits") so it can be reconsidered.
Biting off the whole thing at once. It's good to say "start small and scale up" rather than demanding a giant sweep on day one.
When in doubt, write the one-sentence version of your idea and stop there.