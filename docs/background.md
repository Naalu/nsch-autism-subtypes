# Background: the question, the data, and the methods

Read this first if you are new to the project. It assumes you can read Python and have seen a mean and a standard deviation, and not much else. The README tells you how to run things; this document tells you what they are doing and why.

## 1. The question

Autism is diagnosed from behavior, and the label covers children who are very different from one another. One child may be nonspeaking with epilepsy and an intellectual disability; another may be in a regular classroom with anxiety and ADHD. Clinicians have long suspected there are "subtypes" hiding inside the single diagnosis.

One way to look for them is through the other conditions a child has. Most autistic children have at least one other diagnosis (ADHD, anxiety, speech disorder, and so on), and these do not occur at random: some pairs go together far more often than chance. If there are a few recurring patterns of co-occurring conditions, those patterns might be subtypes. Several research groups have reported such patterns, usually three or four groups, using hospital records or clinic samples.

The problem is that almost none of those studies checked whether their groups show up again in a second, independent set of children. A clustering algorithm will always hand you groups; the question is whether the groups are a property of autism or a property of that particular sample and that particular algorithm.

This project asks the question on a national survey instead of a clinic, and builds the checking in from the start. We decided in advance what "the groups are real" would have to look like, and if the data fail those checks, that is the result we report.

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

Each round makes the model fit a little better, or at least no worse, so it eventually settles. It can settle in a mediocre place depending on the starting guess, so we run it from 20 random starts and keep the best. Writing EM from scratch is a requirement of the course; the StepMix library is used to check our version.

### Choosing K with BIC

The model needs to be told how many classes to look for. More classes always fit the data at least as well, so "best fit" alone would choose the largest K we tried. The Bayesian information criterion (BIC) adds a penalty for each extra parameter; the K with the lowest BIC is the one where the extra classes stop paying for themselves. It is the standard choice in the latent class literature.

On our data BIC kept falling as K grew, at least through 8. That is common with large samples and is one of the open decisions in the project.

### Hierarchical clustering (the proposal's second method)

This method needs no probability model. It needs only a way to measure how far apart two children are. We use the Jaccard distance: the share of conditions that one child has and the other does not, out of all the conditions either has. Two children with identical rows are at distance 0; two with no conditions in common are at distance 1. Conditions that both children lack do not count, which matters here because most children lack most of the 18 conditions.

The algorithm starts with every child as their own group and repeatedly merges the two closest groups, until everyone is in one group. The distance between two groups is the average distance between their members ("average linkage"). Recording the merges gives a tree; cutting the tree gives K groups.

On this data the method behaved badly: it kept splitting off tiny groups of unusual children and leaving one group with almost everyone. That is a known weakness of average linkage with sparse data, and it is why a third method was added.

### k-means (the alternative second method)

k-means picks K "centers", assigns each child to the nearest center, moves each center to the average of its children, and repeats until nothing changes. Distance is the number of conditions on which a child and a center disagree. It is simple, fast, and was used by one of the published studies we compare against. Like EM it depends on the starting centers, so it is also run from many starts.

## 4. The checks

Each check answers a specific skeptical question.

### "Would a different method find the same groups?" (adjusted Rand index)

We cluster the same children with the latent class model and with a second method, then measure how much the two groupings agree. The adjusted Rand index (ARI) does this. It looks at every pair of children and asks whether the two methods agree on whether that pair belongs together, then subtracts what you would expect from random labels. 1 means identical groupings, 0 means no better than chance. It does not care what the groups are called.

The pass line is ARI of at least 0.65. Steinley (2004) studied the index and described 0.65 to 0.80 as moderate agreement, 0.80 to 0.90 as good, above 0.90 as excellent. We took the bottom of "moderate" because the two methods are built on different ideas and we did not expect them to agree perfectly.

### "Would the groups show up in different children?" (held-out years and the 2025 forward test)

We fit a model on the 2016 to 2021 children and another on the 2022 to 2024 children. Then we use both to label the 2022 to 2024 children and compare the labelings with ARI. If a model that never saw those children labels them about the same way as a model fit on them, the groups travel.

The 2025 survey was released on October 5, 2026, after the project's design was fixed. We label the 2025 children with the frozen 2016 to 2024 model, fit a fresh model on 2025 alone, and compare. Nothing could have been tuned to 2025, which makes it the cleanest replication we have.

### "Do the groups depend on a few particular children?" (bootstrap)

Draw a new sample of 8,891 by picking children at random with replacement (some appear twice, some not at all), refit the model, and see how much of each original class comes back together. Repeat 200 times. A class that reappears nearly intact every time is stable; one that falls apart was an accident of the sample. The Jaccard index (shared members over all members) scores each class, and Hennig (2007) gives the reading: 0.75 and above is stable, 0.5 and below has dissolved.

### "Are the groups just low, medium and high?" (count versus class)

The most boring explanation of any subtype result is that the groups are children with few, some, and many conditions, renamed. To test that we use something the model never saw: the parent's rating of how severe the child's autism is (mild, moderate, severe). We fit three models of severity, one using only the child's count of conditions, one using only the child's class, and one using both. If the class does not predict severity any better than the count, the classes add nothing, and we say so.

### "What would no structure look like?" (shuffled baseline)

Shuffle each column of the table independently. Every condition keeps its prevalence, but every link between conditions is destroyed. Fitting the latent class model to the shuffled table shows how the BIC curve behaves when there is nothing to find, for comparison with the real curve.

## 5. A glossary

- ARI (adjusted Rand index): agreement between two clusterings of the same items, 0 for chance and 1 for identical.
- BIC: a model score that rewards fit and penalizes parameters; used to choose the number of classes. Lower is better.
- Bootstrap: resampling the data with replacement to see how much a result depends on the particular sample.
- Class / cluster: a group of children found by a method. "Class" is the latent class word; "cluster" is the general word.
- Co-occurring condition: another diagnosis an autistic child has (ADHD, anxiety, epilepsy, and so on). Also called a comorbidity.
- EM (Expectation-Maximization): the iterative algorithm that fits a latent class model.
- Entropy (relative): how cleanly the model assigns children to classes, 1 for certain, 0 for no idea.
- Flag: a 0/1 column saying whether a child currently has a condition.
- Jaccard: shared items divided by items in either set. Used as a distance between children (1 minus Jaccard) and as a stability score for classes.
- K: the number of classes or clusters.
- Latent class analysis: a model in which hidden classes explain the pattern of 0/1 answers.
- Linkage: in hierarchical clustering, the rule for the distance between two groups. We use average linkage.
- Prevalence: the share of children who have a condition.
- Profile: a class's probability of each condition, as a row of numbers.
- Responsibilities: for each child, the probability of belonging to each class.
- Stratum, weight: survey design variables. The weight says how many U.S. children a sampled child represents; strata are the groups the sample was drawn within, needed for correct standard errors.
- Topical file: the Census Bureau's name for the main NSCH data file for one year.

## 6. Where to go next

- `reports/data-overview.md` shows the actual rows, raw and processed, with figures.
- `docs/validation-checklist.md` lines the project up against a published checklist for subtyping studies and lists the papers we compare with.
- `docs/implementation-spec.md` is the contract for the from-scratch code.
- The scripts in `scripts/` are numbered in the order described here and each starts with a plain-language summary.
