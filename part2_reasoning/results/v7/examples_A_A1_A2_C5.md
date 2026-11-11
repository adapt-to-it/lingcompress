# Esempi v7: stesso problema in A / A1 / A2 / C5

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

**A1**

```
There are 80/100 * 10 = 8 more purple flowers than yellow.
So there are 10 + 8 = 18 purple flowers.
Yellow and purple together total 10 + 18 = 28.
Green flowers are 25/100 * 28 = 7.
Total flowers are 28 + 7 = 35.
The answer is 35.
```

**A2**

```
80/100 * 10 = 8 more purple.
10 + 8 = 18 purple.
10 + 18 = 28 yellow and purple.
25/100 * 28 = 7 green.
28 + 7 = 35.
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

## train-446

**Domanda.** In a classroom, 10 children have 7 books each. Their teacher brings another 8 books to the classroom. How many books are in the classroom altogether?

**A**

```
The children have 10 * 7 = 70 books altogether.
With their teacher bringing 8 more books, the total number of books in the classroom is 70 + 8 = 78.
The answer is 78.
```

**A1**

```
The children have 10 * 7 = 70 books.
The teacher adds 8 more books.
70 + 8 = 78.
The answer is 78.
```

**A2**

```
Children have 10 * 7 = 70.
Teacher adds 8.
Total: 70 + 8 = 78.
The answer is 78.
```

**C5**

```
books_children = 10 * 7 = 70
books_total = 70 + 8 = 78
ANSWER = 78
```

## train-3354

**Domanda.** Mr. Lalande inherited 20,000 euros from his old aunt Adeline. He is very happy because he will be able to afford the car of his dreams, a superb car worth 18000 €. He goes to the dealership and tries the car. He decides to take it, but instead of paying for everything right away, he chooses to pay in several installments. He pays 3,000 euros to be able to leave with the car and the seller offers to let him pay the rest in 6 monthly installments. How much does Mr. Lalande have to pay every month?

**A**

```
The total cost of the car is 18000 euros.
Mr. Lalande pays 3000 euros upfront, so the remaining amount is 18000 - 3000 = 15000 euros.
He will pay this remaining amount in 6 monthly installments, so each installment is 15000 / 6 = 2500 euros.
The answer is 2500.
```

**A1**

```
Car costs 18000 euros.
Mr. Lalande pays 3000 upfront.
Remaining: 18000 - 3000 = 15000.
Divide by 6: 15000 / 6 = 2500.
The answer is 2500.
```

**A2**

```
Car cost: 18000.
Upfront: 3000.
Remaining: 18000 - 3000 = 15000.
Monthly: 15000 / 6 = 2500.
The answer is 2500.
```

**C5**

```
GIVEN cost_of_car = 18000
GIVEN down_payment = 3000
remaining_amount = 18000 - 3000 = 15000
monthly_payment = 15000 / 6 = 2500
ANSWER = 2500
```
