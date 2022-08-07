import numpy as np

state_action_values = np.ones((10, 10, 2, 2))
# initialze counts to 1 to avoid division by 0
state_action_pair_count = np.ones((10, 10, 2, 2)) *2
state_action_pair_count[3, 5, 1, :] = 3

fenzi = state_action_values[2, 4, 0, :]
fenmu = state_action_pair_count[3, 5, 1, :]
values_ =  fenzi / fenmu

print("======values:", values_, " max(values:)", np.max(values_))
values_[1] = 1

tmp = [action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)]
print("tmp:", tmp)

a = [x for x in range(1, 10) if x % 2]

b = [ x if x%2 else x*100 for x in range(1, 10) ]
print(a,b)

# One-line list comprehension: if-else variants
# https://stackoverflow.com/questions/17321138/one-line-list-comprehension-if-else-variants

# One Line If-Else Statements in Python
# Writing a one-line if-else statement in Python is possible by using the ternary operator, also known as the conditional expression.

# Here is the syntax of the one-liner ternary operator:

# some_expression if condition else other_expression