import sys
import math


def get_parameter_vectors():

    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    X = {chr(65+i): 0 for i in range(26)}  # Initialize counts for A-Z
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            for char in line:
                upper_char = char.upper()
                if upper_char in X:
                    X[upper_char] += 1
    return X


def compute_F(counts, probs, prior):
    F = math.log(prior)
    for i in range(26):
        c = chr(ord('A') + i)
        X_i = counts.get(c, 0)
        p_i = probs[i]
        if p_i > 0:
            F += X_i * math.log(p_i)
        else:
            if X_i != 0:
                F += X_i * float('-inf')
    return F

# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python3 hw2.py [letter file] [english prior] [spanish prior]")
        sys.exit(1)

    letter_file = sys.argv[1]
    prior_e = float(sys.argv[2])
    prior_s = float(sys.argv[3])

    e, s = get_parameter_vectors()
    counts = shred(letter_file)

    # Q1
    print('Q1')
    for i in range(26):
        c = chr(ord('A') + i)
        count = counts.get(c, 0)
        print(f'{c} {count}')

    # Q2
    print('Q2')
    X1 = counts.get('A', 0)
    e1 = e[0]
    s1 = s[0]

    if e1 > 0:
        val_e = X1 * math.log(e1)
    else:
        val_e = float('-inf') if X1 != 0 else 0.0

    if s1 > 0:
        val_s = X1 * math.log(s1)
    else:
        val_s = float('-inf') if X1 != 0 else 0.0

    print(f"{val_e:.4f}")
    print(f"{val_s:.4f}")

    # Q3
    print('Q3')
    F_english = compute_F(counts, e, prior_e)
    F_spanish = compute_F(counts, s, prior_s)

    print(f"{F_english:.4f}")
    print(f"{F_spanish:.4f}")

    # Q4
    print('Q4')
    D = F_spanish - F_english

    if D >= 100:
        prob_english = 0.0
    elif D <= -100:
        prob_english = 1.0
    else:
        prob_english = 1 / (1 + math.exp(D))

    print(f"{prob_english:.4f}")