Frequently asked questions
The questions people ask in their first hours with Primus. For the longer version of the prompting advice, see Writing research prompts.

Getting started
I'm a Machine Learning Engineer. How can Primus benefit me?
If you are a machine learning engineer, Primus can complement your team and help you explore more research paths, quickly. For example, you can use Primus to act like a team of interns: it can conduct literature reviews, set up experiments, and babysit clusters—tasks you would otherwise have to do manually.

Are there new techniques in the industry that you'd like to explore? Are there opportunities to optimize a model or inference engine in your current pipeline? Do you have an idea you'd like to explore? Primus is great at taking over the time-intensive parts of these tasks on your behalf.

Do I need to have a background in Machine Learning to use Primus?
Many of Primus's users are not machine learning engineers. Primus can help guide you and re-formulate your questions to become valid experiments based on industry and academic best-practices.

You don't need a dataset ready, a model picked out, or a clear idea of what experiment to run. Primus can do that work, and shows each step as it goes.

Knowing a subfield well helps you judge the result once it lands.

How do I get access?
Create an account at primus.lab.cloud. Primus, by default, runs on our internal and partner GPU clusters. But our GPU capacity is limited, so we are onboarding users from the waitlist in waves as GPU capacity becomes available. Join it here and we'll bring you in as soon as a spot opens.

If you're an organization that wants Primus for your team, get in touch.

What do I write in my first prompt?
Something you're curious about; vague is fine. One prompt that started a project:

"I think LLMs might hallucinate less if they were aware of their own token probabilities. Help me figure out an experiment around that."

Primus asks follow-up questions and sharpens the project with you before the work starts.

The one habit worth having: aim narrow rather than grand. Name a specific failure and try to improve it without breaking everything else. Writing research prompts goes deeper, including nine question shapes that have worked.

Is there a size limit on documents I upload with a prompt?
Yes, 25 MB per document. Documents you attach to a new prompt are meant to give Primus more context about your project: a paper, a spec, notes on prior work. They aren't meant to carry the data itself. If you're bumping into the limit, you're probably trying to upload a dataset or model, which has a better path (see below).

How do I give Primus a large dataset or model?
Upload it somewhere Primus can reach, such as cloud storage (S3, Google Cloud Storage) or a hub like Hugging Face. Then add the access token in User settings → Secrets and steer Primus to access the data from there. Primus pulls the data down when the project needs it.

We're building in-house storage for datasets and models so you can hand them to Primus directly. Coming soon.

Running a project
Does Primus wait for my approval, or does it just keep going?
Primus can operate in two modes, configurable in user settings. Full Self-Driving is on by default, this is where Primus makes its own routine sign-offs and carries on working while you're doing something else.

It still stops for you in three situations:

A blocker it can't get past on its own.
Spending past the compute budget you approved.
A rescope, when it wants to change what the project is about.
If you'd rather approve each step yourself, turn Full Self-Driving off in settings for every project, or in a single project's settings to slow just that one down.

How will I know when it needs me?
Primus emails you when a run parks on a question and when it finishes or fails. It won't send more than one email per event, and progress updates don't generate mail at all. If you connect Slack, the same two moments post there.

Can I steer it while it's running?
Yes, you can chat with Primus while any project is open. Each project has a live workspace where you read findings as they land and say what you want changed: defer that comparison, focus on the smaller model first, stop tuning and write it up. Primus applies that to the next thing it does.

How long does a project take?
Because Primus projects run in the physical world, on real GPUs, project length can vary widely based on the experiment. A tight comparison can finish in hours; an open-ended one runs for days.

Progress is visible the whole time, and Primus asks when it needs a decision from you.

Can I run more than one project at once?
Yes. Separate projects are isolated and can run at the same time. Your plan may limit the total number of projects you can run.

Compute and cost
What does it cost?
We are offering free trial access to Primus where the class of GPUs accessible to the free tier is limited (in order to control cost and ensure we can obtain capacity). If you are interested in upgrading to a paid tier (which gives you access to larger GPUs and more projects) please contact us.

Do I need my own GPUs?
By default, Primus runs its experiments on GPUs that we obtain from our cloud GPU partners.

Organizations that would rather run on their own compute can do that on our enterprise plan. Get in touch.

How can I prevent overspending on GPU costs?
A compute budget is set as part of the research plan, and approving that plan is where you approve the cap. From then on the cap is enforced: a run that would spend past it stops and asks you.

Each project also has its own budget and compute views, so while the work is still going you can see what has been spent, on which experiment, and at which stage.

Results and sharing
What do I get at the end?
A finished paper — methods, results, and citations.
A model card — what was trained, on what, and how it behaves.
A reproducibility package — enough to re-run the work and check it.
Trained models and artifacts — checkpoints, datasets, and code, with a record of how each was made.
A negative result, when that's the result. If the hunch doesn't hold up, the paper says so and shows why.
Examples of finished work are on our Research page.

Restrictions
Is Primus restricted from certain types of research?
Yes. Primus is equipped with safety guardrails that prevent it from engaging in high-risk or prohibited domains. This includes generating novel pathogens, developing zero-day software exploits, or designing weaponry. Please refer to our Terms of Service and Acceptable Use Policy for a complete list of prohibited research areas.

How do I know Primus isn't fabricating data?
Primus, like all AI, can hallucinate. However, it is designed to actively reduce this possibility. Primus works to ground its facts and applies internal critics to force itself to be rigorous, ensuring all claims are grounded against reproducible experiments. The metrics and outcomes it reports are pulled directly from experiments that are run in the physical world. To ensure total transparency, every finished project includes a reproducibility package—containing the raw logs, code, and datasets—so you can independently verify every claim the final paper makes.

Can I submit articles produced by Primus to journals and conferences?
No. Our Terms of Service explicitly prohibit this. The academic peer-review system relies entirely on the limited, unpaid time of volunteer experts. It is not fair to burden human reviewers by asking them to evaluate research written by AI. As stated in our Terms under Respect Peer Reviewers' Time: "you explicitly agree not to submit AI-generated papers, manuscripts, or research authored by Primus to journals, conferences, or peer-reviewed academic venues."

Who owns the discoveries and intellectual property?
You do. The research directions you prompt, along with the resulting models, code, artifacts, and discoveries, belong entirely to you or your organization. We do not claim ownership over the outputs of your projects.

Can I use Primus for research involving human data?
Primus can process and analyze datasets you provide, but it cannot navigate ethical compliance for you. If your research involves human subjects, Personally Identifiable Information (PII), or clinical data, you are responsible for obtaining the necessary Institutional Review Board (IRB) or ethics committee approvals. You must also ensure any data is properly anonymized and legally cleared before bringing it into a Primus workspace.

Will Primus cite the work of human researchers fairly?
Yes. During the literature review phase, Primus identifies and reads relevant academic papers to ground its hypotheses. It is instructed to properly cite these sources in the final paper it delivers, adhering to standard academic attribution practices, just as a human researcher would.

Getting help
I'm stuck. Where do I go?
Email hello@lab.cloud or join our Discord to get help from the Primus team and the community.