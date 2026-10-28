# Background: the question, the data, and the methods

Read this first if you are new to the project. It assumes you can read Python and have seen a mean and a standard deviation, and not much else. The README tells you how to run things; this document tells you what they are doing and why.

## 1. The question

Autism is diagnosed from behavior, and the label covers children who are very different from one another. One child may be nonspeaking with epilepsy and an intellectual disability; another may be in a regular classroom with anxiety and ADHD. Clinicians have long suspected there are "subtypes" hiding inside the single diagnosis.

One way to look for them is through the other conditions a child has. Most autistic children have at least one other diagnosis (ADHD, anxiety, speech disorder, and so on), and these do not occur at random: some pairs go together far more often than chance. If there are a few recurring patterns of co-occurring conditions, those patterns might be subtypes. Several research groups have reported such patterns, usually three or four groups, using hospital records or clinic samples.

The problem is that almost none of those studies checked whether their groups show up again in a second, independent set of children. A clustering algorithm will always hand you groups; the question is whether the groups are a property of autism or a property of that particular sample and that particular algorithm.

This project asks the question on a national survey instead of a clinic, and builds the checking in from the start. We decided in advance what "the groups are real" would have to look like, and if the data fail those checks, that is the result we report. We also fit a model with no groups at all, in which children vary smoothly along a few continuous scales, and ask which description the data prefer.

## 2. The data

### What the survey is

The National Survey of Children's Health (NSCH) is run every year by the U.S. Census Bureau for the Health Resources and Services Administration. Households are chosen at random by address. One child in the household is picked, and a parent or caregiver fills out a questionnaire about that child, online or on paper. Between 20,000 and 55,000 children are covered each year, and the sample is designed so that, with the right weights, it represents all children in the United States.

We use the ten yearly files from 2016 through 2025. They are public and free to download; nobody has to apply for access.

### What we take from it

For each child the survey asks, for about two dozen conditions: "Has a doctor or other health care provider ever told you that this child has [condition]?" and, if yes, "Does this child currently have the condition?" We use 18 of those conditions (the config file explains why the others are out) and turn each pair of answers into a single flag: 1 if the child currently has it, 0 if not.

We keep children aged 6 to 17 whose parent says they currently have autism. That gives 9,710 children for 2016 to 2024 and 1,660 for 2025. After dropping children with a blank for any of the 18 conditions, 8,891 and 1,505 remain.

So the object every method works on is a table of 8,891 rows (children) by 18 columns (conditions), containing only 0s and 1s. Everything else in the project is about finding and checking structure in that table.

### Things to keep in mind about survey data

The flags are what a parent reports a doctor said. A child whose family cannot get to a specialist may have an undiagnosed condition and a 0 in our table. The flags therefore mix "has the condition" with "has had access to care".

Each child appears once. We can describe groups of children, but we cannot say whether a child moves from one group to another as they grow.

The survey is weighted. Some states and some kinds of households are deliberately oversampled, and each child carries a weight saying how many U.S. children they stand for. We use the weights only at the end, to say how common each group is nationally. We do not use them when finding the groups.

## 3. The methods, one at a time

### Clustering, and what "unsupervised" means

In most machine learning you have a right answer for each example and teach a model to predict it. Here there is no right answer: nobody has labeled the children with their subtype. Clustering is the family of methods that look for groups in data without being told what the groups are. "Unsupervised" means there is no answer key.

Because there is no answer key, there is also no accuracy score. That is why so much of this project is about checking.

### Latent class analysis (the main method)

Imagine there are K hidden types of children. Each type has its own probability of having each condition. Type A might have ADHD 90% of the time and epilepsy 5% of the time; type B the other way round. We never observe a child's type. We only see the row of 0s and 1s, and we ask: which K types, with which probabilities, would most plausibly have produced all 8,891 rows?

That is a latent class model ("latent" meaning hidden). In statistics it is a mixture of Bernoulli distributions. Its outputs are:

- the size of each class (what fraction of children belong to it);
- the condition profile of each class (the probability of each of the 18 conditions, given the class);
- for each child, the probability of belonging to each class.

The last one matters. The model does not force a child into a box. A child can be 70% class 1 and 30% class 3, and we carry that uncertainty through to the end.

### EM: how the model is fit

There is no formula that gives the best class sizes and profiles directly, so we use an iterative method called Expectation-Maximization. Start with a random guess for the profiles. Then alternate:

1. E-step. Given the current profiles, work out for each child how likely each class is.
2. M-step. Given those likelihoods, update each class's profile to be the (weighted) average of the children who probably belong to it, and update each class's size.

Each round makes the model fit a little better, or at least no worse, so it eventually settles. It can settle in a mediocre place depending on the starting guess, so we run it from 20 random starts and keep the best. Writing EM from scratch is a requirement of the course, and the StepMix library is used to check our version. Until our version passes its tests (`docs/implementation-spec.md`), the numbers reported in this repository come from StepMix.

### Choosing K

The model needs to be told how many classes to look for. More classes always fit the data at least as well, so "best fit" alone would choose the largest K we tried. The Bayesian information criterion (BIC) adds a penalty for each extra parameter; the K with the lowest BIC is the one where the extra classes stop paying for themselves. It is the standard choice in the latent class literature.

On our data BIC kept falling as K grew, at least through 8, so it would pick whatever upper limit we set. That is common with large samples, and it is also what happens when the real structure is continuous instead of grouped (section 4 explains why). We therefore needed a different rule, and we chose it after seeing that BIC did not settle. The paper says so.

The rule we use is stability: the largest K at which every class survives resampling (the bootstrap check below, mean Jaccard of at least 0.75). The idea, from Ben-Hur, Elisseeff and Guyon (2002) and Lange et al. (2004), is that a real group should come back when you redraw the sample, and a group that only appears in some resamples is not one you can describe to anyone. On our data that rule gives K = 4. Choosing K this way uses only the bootstrap, so the other checks (a second method, later survey years, age bands) remain independent tests of the K = 4 classes rather than inputs to choosing them.

One thing stability cannot do is tell groups from a continuum. Picture children spread evenly over a disc with no groups at all: "cut the disc into four quarters" gives the same quarters in every resample, so those four "classes" are perfectly stable. We checked this on simulated data with no groups, and classes stayed stable up to K = 5. So K = 4 is best read as "the finest division of these children that comes back reliably". Whether the pieces are real groups is a separate question, answered by the checks in section 4 that compare classes with continuous scales.

### Hierarchical clustering (the proposal's second method)

This method needs no probability model. It needs only a way to measure how far apart two children are. We use the Jaccard distance: the share of conditions that one child has and the other does not, out of all the conditions either has. Two children with identical rows are at distance 0; two with no conditions in common are at distance 1. Conditions that both children lack do not count, which matters here because most children lack most of the 18 conditions.

The algorithm starts with every child as their own group and repeatedly merges the two closest groups, until everyone is in one group. The distance between two groups is the average distance between their members ("average linkage"). Recording the merges gives a tree; cutting the tree gives K groups.

On this data the method behaved badly: it kept splitting off tiny groups of unusual children and leaving one group with almost everyone. This is called chaining. Average linkage joins any two groups that are close on average, and with 0/1 data there are thousands of children at small distances from one another, so one group keeps absorbing its neighbors. It is a known weakness of average linkage on sparse binary data, and it is why k-means was added as the second method.

### k-means (the alternative second method)

k-means picks K "centers", assigns each child to the nearest center, moves each center to the average of its children, and repeats until nothing changes. Distance is the number of conditions on which a child and a center disagree. It is simple, fast, and was used by one of the published studies we compare against. Like EM it depends on the starting centers, so it is also run from many starts.

On this data k-means has many solutions that are almost as good as each other (their total distances differ by less than 0.1%) but split the children differently. Which one you get depends on the random seed. So we run it from 200 starts with each of five seeds, report the best, and also report how much the answer moves from seed to seed. When an algorithm cannot make up its mind like this, it usually means there are no clean gaps between groups for it to find.

## 4. The checks

Each check answers a specific skeptical question.

### "Would a different method find the same groups?" (adjusted Rand index)

We cluster the same children with the latent class model and with a second method, then measure how much the two groupings agree. The adjusted Rand index (ARI) does this. It looks at every pair of children and asks whether the two methods agree on whether that pair belongs together, then subtracts what you would expect from random labels. 1 means identical groupings, 0 means no better than chance. It does not care what the groups are called.

The pass line is ARI of at least 0.65. Steinley (2004) studied the index and described 0.65 to 0.80 as moderate agreement, 0.80 to 0.90 as good, above 0.90 as excellent. We took the bottom of "moderate" because the two methods are built on different ideas and we did not expect them to agree perfectly.

### "Would the groups show up in different children?" (held-out years and the 2025 forward test)

We fit a model on the 2016 to 2021 children and another on the 2022 to 2024 children. Then we use both to label the 2022 to 2024 children and compare the labelings with ARI. If a model that never saw those children labels them about the same way as a model fit on them, the groups travel.

The 2025 survey was released on October 5, 2026, eight days before the project proposal fixed the design. We processed it with the other years and summarized it in the data overview, but did not fit anything to it until this check. We label the 2025 children with the frozen 2016 to 2024 model, fit a fresh model on 2025 alone, and compare. No modeling choice was made on 2025, which makes it the cleanest replication we have.

### "Do the groups depend on a few particular children?" (bootstrap)

Draw a new sample of 8,891 by picking children at random with replacement (some appear twice, some not at all), refit the model, and see how much of each original class comes back together. Repeat 200 times. A class that reappears nearly intact every time is stable; one that falls apart was an accident of the sample. The Jaccard index (shared members over all members) scores each class, and Hennig (2007) gives the reading: 0.75 and above is stable, 0.5 and below has dissolved.

### "Are the groups just low, medium and high?" (count versus class)

The most boring explanation of any subtype result is that the groups are children with few, some, and many conditions, renamed. To test that we use something the model never saw: the parent's rating of how severe the child's autism is (mild, moderate, severe). We fit three models of severity, one using only the child's count of conditions, one using only the child's class, and one using both. If the class does not predict severity any better than the count, the classes add nothing, and we say so.

The rule as written before fitting was "if class and both do not beat count, the classes are a gradient in disguise." That sentence can be read two ways, and we report both: count plus class does beat count alone, and class alone does not. We also compare against other summaries of the same 18 flags: each child's class probabilities instead of a single label, each child's position on the continuous scales described below, and all 18 flags entered separately. Those comparisons say how much of what severity tracks is captured by the classes compared with the alternatives.

### "What would no structure look like?" (shuffled baseline)

Shuffle each column of the table independently. Every condition keeps its prevalence, but every link between conditions is destroyed. Fitting the latent class model to the shuffled table shows how the BIC curve behaves when there is nothing to find, for comparison with the real curve.

### "Are the groups an age effect?" (age bands)

Older children are diagnosed with more anxiety and depression, younger ones with more developmental delay. If the classes were just "younger" and "older", they would fall apart when fit within a narrow age range. We refit the model separately on ages 6 to 11 and 12 to 17 and compare each band's classes with the pooled classes on the same children (ARI, pass line 0.65).

### "Do the groups depend on how a condition is counted?" (parallel items)

Our flags use "currently has". The survey also asks "ever told", which counts conditions a child has outgrown. We rebuild every flag from "ever told", refit, and compare with the main classes. If the classes depended on exactly where the "currently" line falls, this would break them.

### "Do the modeling choices matter?" (robustness)

Three smaller checks: refitting with the survey weights (we normally fit unweighted), comparing the children dropped for a blank answer with those kept, and refitting from five other random seeds.

### "Groups, or scales?" (the continuous model)

A latent class model says each child belongs to one of K hidden types. The alternative is that there are no types, only a few hidden scales: each child sits somewhere on, say, a "developmental" scale and a "psychiatric" scale, and the higher a child sits on a scale, the more likely the conditions tied to that scale. This is a latent trait model (also called binary factor analysis or multidimensional item response theory). With D scales it has one slope per condition per scale, saying how strongly that condition depends on that scale, and one intercept per condition. Each child gets a score on each scale instead of a class.

Both models give a likelihood for the same 0/1 table, so they can be compared directly. Three comparisons are used (`scripts/12_dimensions.py`):

1. Fit. BIC for latent trait models with 1 to 4 scales against latent class models with 1 to 8 classes.
2. Pairs of conditions. Both models assume the conditions are unrelated once you know a child's class (or scores). For every pair of conditions we compare how often the pair occurs together with how often the model says it should (a "bivariate residual"). A model that misses many pairs is leaving structure on the table, and a class model will add classes to soak it up.
3. Simulation. We generate fake data from the fitted continuous model, which has no groups in it by construction, and run the latent class analysis on it. If the class analysis on fake continuous data looks like the class analysis on the real data (BIC still falling at K = 8, similar entropy), then the real BIC curve is not evidence of groups. We also do the reverse, generating data from the four-class model, to check that the comparison can tell the two apart. Bauer and Curran (2004) showed that latent class models find "classes" in continuous data, which is why this check is needed.

### "Do children clump together?" (the shape of the cloud)

The continuous model assumes children's positions on its two scales follow a bell curve: most children near the middle, fewer toward the edges, no clumps. If the children really fell into four groups, their positions would pile up in four places. `scripts/14_shape_and_stability.py` estimates the distribution of positions from the data instead of assuming it, and measures how much better that estimate explains the children than the bell curve does (in log-likelihood units, where bigger means a stronger departure from a bell curve).

The estimate is always a bit lumpy, even when the truth is a smooth bell curve, so the number only means something next to a yardstick. We make fake datasets of the same size from the continuous model (no clumps) and from the four-class model (four tight clumps), run each through the same estimate, and see where the real data fall between them.

### "How much structure generalizes?" (held-out likelihood)

BIC is computed on the same children the model was fit on. A sharper question is how well a model fit on some years predicts children it has never seen. `scripts/13_heldout_selection.py` fits every latent class and latent trait model on 2016 to 2021 and scores it on 2022 to 2024, then fits on 2016 to 2024 and scores it on 2025. The score is the held-out log-likelihood: how probable the model finds the new children's rows. Higher is better, and it cannot be gamed by adding parameters, because extra parameters that only fit noise make predictions on new children worse.

This answers "how much structure is there, and which kind of model captures it more economically", not "which K is right". On our data the held-out score keeps improving through K = 8, and a two-scale model with 53 parameters predicts new children about as well as eight classes with 151. That is the strongest single piece of evidence that the structure is mostly continuous.

## 5. A glossary

- ARI (adjusted Rand index): agreement between two clusterings of the same items, 0 for chance and 1 for identical.
- Bivariate residual: for one pair of conditions, how far the observed number of children with both is from what a model predicts. Values above 3.84 flag a pair the model fits poorly.
- BIC: a model score that rewards fit and penalizes parameters; used to choose the number of classes. Lower is better.
- Bootstrap: resampling the data with replacement to see how much a result depends on the particular sample.
- Class / cluster: a group of children found by a method. "Class" is the latent class word; "cluster" is the general word.
- Co-occurring condition: another diagnosis an autistic child has (ADHD, anxiety, epilepsy, and so on). Also called a comorbidity.
- EM (Expectation-Maximization): the iterative algorithm that fits a latent class model.
- Entropy (relative): how cleanly the model assigns children to classes, 1 for certain, 0 for no idea.
- Factor score: a child's estimated position on each scale of a latent trait model.
- Flag: a 0/1 column saying whether a child currently has a condition.
- Held-out log-likelihood: how probable a model finds data it was not fit on. Higher is better.
- Jaccard: shared items divided by items in either set. Used as a distance between children (1 minus Jaccard) and as a stability score for classes.
- K: the number of classes or clusters.
- Latent class analysis: a model in which hidden classes explain the pattern of 0/1 answers.
- Latent trait model: a model in which a few hidden continuous scales explain the pattern of 0/1 answers. The continuous counterpart of latent class analysis.
- Linkage: in hierarchical clustering, the rule for the distance between two groups. We use average linkage.
- Prevalence: the share of children who have a condition.
- Profile: a class's probability of each condition, as a row of numbers.
- Plausible value: one random draw of a child's position from the range of positions consistent with their answers. Used for plotting children in the continuous model.
- Responsibilities: for each child, the probability of belonging to each class.
- Slope (or loading): in a latent trait model, how strongly a condition depends on a scale.
- Stability (cluster stability): how reliably a class reappears when the sample is redrawn; scored with the bootstrap Jaccard.
- Stratum, weight: survey design variables. The weight says how many U.S. children a sampled child represents; strata are the groups the sample was drawn within, needed for correct standard errors.
- Topical file: the Census Bureau's name for the main NSCH data file for one year.

## 6. Where to go next

- `reports/data-overview.md` shows the actual rows, raw and processed, with figures.
- `docs/validation-checklist.md` lines the project up against a published checklist for subtyping studies and lists the papers we compare with.
- `docs/implementation-spec.md` is the contract for the from-scratch code.
- `docs/outputs.md` lists every file the pipeline writes and how to add a condition or a check.
- The scripts in `scripts/` are numbered in the order described here and each starts with a plain-language summary.
