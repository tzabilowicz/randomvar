def factorial(x):
    if x == 0:
        return 1
    else:
        return x * factorial(x-1)
    
def choose(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return factorial(n) // (factorial(k) * factorial(n - k))