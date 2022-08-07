
# Python program to illustrate
# enumerate function in loops
l1 = ["eat", "sleep", "repeat"]
  
# printing the tuples in object directly
for ele in enumerate(l1):
    print (ele)
  
# changing index and printing separately
for count, ele in enumerate(l1, 100):
    print (count, ele)
  
# getting desired output from tuple
for count, ele in enumerate(l1):
    print(count)
    print(ele)


print("======================================================")
td_alphas = [0.15, 0.1, 0.05]
mc_alphas = [0.01, 0.02, 0.03, 0.04]
print(td_alphas + mc_alphas)
td_alphas_1 = [0.15, 0.1, 1.0]
print(td_alphas + td_alphas_1)

for i, alpha in enumerate(td_alphas + mc_alphas): # 加好可以这样用?????
    print("---- i:",i, " alpha:", alpha)