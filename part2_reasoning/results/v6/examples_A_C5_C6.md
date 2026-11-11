# Esempi v6: stesso problema in A / C5 / C6

## train-5

**Domanda.** Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple. There are only 25% as many green flowers as there are yellow and purple flowers. How many flowers does Mark have in his garden?

**A**

```
There are 80/100 * 10 = 8 more purple flowers than yellow flowers.
So in Mark's garden, there are 10 + 8 = 18 purple flowers.
Purple and yellow flowers sum up to 10 + 18 = 28 flowers.
That means in Mark's garden there are 25/100 * 28 = 7 green flowers.
So in total Mark has 28 + 7 = 35 plants in his garden.
The answer is 35.
```

**C5**

```
GIVEN yellow_flowers = 10
purple_excess = 80 / 100 * 10 = 8
purple_flowers = 10 + 8 = 18
yellow_plus_purple = 10 + 18 = 28
green_flowers = 25 / 100 * 28 = 7
total_flowers = 28 + 7 = 35
ANSWER = 35
```

**C6**

```
purple_extra_n = 10 * 80 / 100 = 8
purple_flowers_n = 10 + 8 = 18
yellow_purple_n = 10 + 18 = 28
green_flowers_n = 28 * 25 / 100 = 7
total_flowers_n = 28 + 7 = 35
ANSWER = 35
```

## train-539

**Domanda.** A paper company decides to operate their business more sustainably. They decide that for every tree they chop down, they want to plant three more. If the company chops down 200 trees in the first half of the year and 300 more trees in the second half of the year, how many more trees does the company need to plant?

**A**

```
The company chopped down 200 + 300 = 500 trees throughout the year.
For every tree they chop, they plant 3 trees, so they need to plant 500 * 3 = 1500 trees.
The answer is 1500.
```

**C5**

```
trees_chopped = 200 + 300 = 500
trees_to_plant = 500 * 3 = 1500
ANSWER = 1500
```

**C6**

```
trees_chopped_n = 200 + 300 = 500
trees_to_plant_n = 500 * 3 = 1500
ANSWER = 1500
```

## train-4202

**Domanda.** For every black & white cookie that Elsa makes, she has to make the batter, bake the cookies for 15 minutes and then allow to cool.  She then has to dip the cookies in white icing and allow the icing to harden for 30 minutes. Then she has to re-dip them in chocolate icing and allowed to harden for an additional 30 minutes.  If it takes her 2 hours to make the cookies from start to finish, how long does it take to make the dough and cool the cookies before she can start dipping them?

**A**

```
It takes 15 minutes to bake, 30 minutes for the white icing to harden, and 30 minutes for the chocolate icing to harden, so 15 + 30 + 30 = 75 minutes for baking and icing.
Every hour has 60 minutes, and it takes her 2 hours to make the cookies from start to finish, so 60 * 2 = 120 minutes total.
The total time is 120 minutes, and 75 minutes are spent on baking and icing, so it takes 120 - 75 = 45 minutes to make the batter and cool the cookies.
The answer is 45.
```

**C5**

```
GIVEN bake_time = 15
GIVEN white_icing_harden = 30
GIVEN chocolate_icing_harden = 30
total_dipping_and_baking = 15 + 30 + 30 = 75
total_time = 60 * 2 = 120
dough_and_cool_time = 120 - 75 = 45
ANSWER = 45
```

**C6**

```
GIVEN bake_time_min = 15
GIVEN white_icing_harden_min = 30
GIVEN chocolate_icing_harden_min = 30
icing_and_bake_time_min = 15 + 30 + 30 = 75
total_time_min = 60 * 2 = 120
dough_and_cool_time_min = 120 - 75 = 45
ANSWER = 45
```
