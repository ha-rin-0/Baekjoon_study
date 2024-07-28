#my code
x=int(input())
y=int(input())
a=y//100
b=(y-a*100)//10
c=(y-a*100-b*10)
print(x*c)
print(x*b)
print(x*a)
print(x*c+x*b*10+x*a*100)

#참고해서 고친 code
a,b=int(input()),int(input())
bb=b
while b:
    print(a*(b%10))
    b//=10
print(a*bb)


#baekjoon 2588