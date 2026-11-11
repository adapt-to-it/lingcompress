# Esempi v5: stesso problema, quattro formati

## train-5

**Domanda.** Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple. There are only 25% as many green flowers as there are yellow and purple flowers. How many flowers does Mark have in his garden?

**A (pieno)**

```
There are 80/100 * 10 = 8 more purple flowers than yellow flowers.
So in Mark's garden, there are 10 + 8 = 18 purple flowers.
Purple and yellow flowers sum up to 10 + 18 = 28 flowers.
That means in Mark's garden there are 25/100 * 28 = 7 green flowers.
So in total Mark has 28 + 7 = 35 plants in his garden.
The answer is 35.
```

**B (telegrafico)**

```
There 80/100 * 10 = 8 more purple flowers than yellow flowers.
So Mark's garden, there 10 + 8 = 18 purple flowers.
Purple and yellow flowers sum up 10 + 18 = 28 flowers.
That means Mark's garden there 25/100 * 28 = 7 green flowers.
So total Mark 28 + 7 = 35 plants his garden.
answer 35.
```

**C (canonico)**

```
GIVEN yellow_flowers = 10
purple_excess = 80 / 100 * 10 = 8
purple_flowers = 10 + 8 = 18
yellow_plus_purple = 10 + 18 = 28
green_flowers = 25 / 100 * 28 = 7
total_flowers = 28 + 7 = 35
ANSWER = 35
```

**D (solo risposta)**

```
35
```

## train-439

**Domanda.** Timothy has $50 to spend at a souvenir shop. He sees some t-shirts that cost $8 each, key chains that sell 3 pieces for $2, and bags that cost $10 each. Timothy buys 2 t-shirts and 2 bags. How many pieces of key chains can he buy with the amount of money he has left?

**A (pieno)**

```
The two t-shirts cost $8 * 2 = $16.
The two bags cost $10 * 2 = $20.
Timothy spent $16 + $20 = $36 on t-shirts and bags.
He has $50 - $36 = $14 left for key chains.
Key chains cost $2 per set, so he can buy $14 / $2 = 7 sets of key chains.
Each set has 3 pieces, so he can buy 7 * 3 = 21 pieces of key chains.
The answer is 21.
```

**B (telegrafico)**

```
two t-shirts cost $8 * 2 = $16.
two bags cost $10 * 2 = $20.
Timothy spent $16 + $20 = $36 t-shirts and bags.
$50 - $36 = $14 left key chains.
Key chains cost $2 per set, so buy $14 / $2 = 7 sets key chains.
Each set 3 pieces, so buy 7 * 3 = 21 pieces key chains.
answer 21.
```

**C (canonico)**

```
cost_two_tshirts = 8 * 2 = 16
cost_two_bags = 10 * 2 = 20
total_spent = 16 + 20 = 36
money_left = 50 - 36 = 14
sets_of_key_chains = 14 / 2 = 7
pieces_of_key_chains = 7 * 3 = 21
ANSWER = 21
```

**D (solo risposta)**

```
21
```

## train-3327

**Domanda.** Tom spends $250 to buy gems in a game he plays.  The game gives 100 gems for each dollar you spend.  Since he bought so many gems he got a 20% bonus of more gems. How many gems did he end up with?

**A (pieno)**

```
Tom spent $250, and the game gives 100 gems for each dollar, so he bought 250 * 100 = 25000 gems.
He received a 20% bonus on the gems, so the bonus is 25000 * 0.2 = 5000 gems.
Adding the bonus to the original gems, he ended up with 25000 + 5000 = 30000 gems.
The answer is 30000.
```

**B (telegrafico)**

```
Tom spent $250, and game gives 100 gems each dollar, so bought 250 * 100 = 25000 gems.
received 20% bonus gems, so bonus 25000 * 0.2 = 5000 gems.
Adding bonus original gems, ended up 25000 + 5000 = 30000 gems.
answer 30000.
```

**C (canonico)**

```
gems_bought = 250 * 100 = 25000
gem_bonus = 25000 * 0.2 = 5000
gems_total = 25000 + 5000 = 30000
ANSWER = 30000
```

**D (solo risposta)**

```
30000
```
