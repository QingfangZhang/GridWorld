def modify(num):
    num += 1
    print(f'num in the func: {num}')


num = 1
modify(num)
print(f'num outside the func: {num}')


def modify(lst):
    lst.pop()
    print(f'lst in the func: {lst}')


my_lst = [1, 2, 3]
modify(my_lst)
print(f'num outside the func: {my_lst}')