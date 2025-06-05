import random  
from datetime import datetime

# This script does some random stuff but is a terrible example
def addNumbers(a,b):
    return a+b

def multiplyNumbers(a,b):
  return  a *b   # inconsistent spacing

def unusedFunction():
    x = 10
    return x

def process_data(data):
    result = []
    for item in data:
       if item%2==0:
            result.append(item**2) 
       else:
            result.append(item * 3)   
    return result

def long_line_function(a, b, c, d, e, f, g, h, i, j, k, l_variable):
 print("This is a very long line that exceeds typical line-length limits and has no real purpose or clarity whatsoever")  # noqa

def read_file(filepath):
    with open(filepath, "r") as f:
        data = f.read()
    return data

def write_file(filepath, content):
    f = open(filepath, "w")
    f.write(content)
    f.close()

def main():
    # missing docstring, unused variables, and random logic
    data_list = [1,2,3,4,5,6,7,8,9,10]
    process_data(data_list)
    random_number = random.randint(1,100)
    print(f"Random number: {random_number}")
    now = datetime.now()  
    print("Current time is: " + str(now))
    content = read_file("nonexistent_file.txt")  # potential FileNotFoundError
    write_file("output.txt", content)
    total = addNumbers(5,   7 )
    product= multiplyNumbers(5,7)
    print("Sum:",total,"Product:",product)

if __name__=="__main__":
    main()  
