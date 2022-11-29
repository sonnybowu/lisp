#!/usr/bin/env python3

import sys
import doctest

sys.setrecursionlimit(10_000)

# NO ADDITIONAL IMPORTS!

#############################
# Scheme-related Exceptions #
#############################


class SchemeError(Exception):
    """
    A type of exception to be raised if there is an error with a Scheme
    program.  Should never be raised directly; rather, subclasses should be
    raised.
    """

    pass


class SchemeSyntaxError(SchemeError):
    """
    Exception to be raised when trying to evaluate a malformed expression.
    """

    pass


class SchemeNameError(SchemeError):
    """
    Exception to be raised when looking up a name that has not been defined.
    """

    pass


class SchemeEvaluationError(SchemeError):
    """
    Exception to be raised if there is an error during evaluation other than a
    SchemeNameError.
    """

    pass


############################
# Tokenization and Parsing #
############################


def number_or_symbol(x):
    """
    Helper function: given a string, convert it to an integer or a float if
    possible; otherwise, return the string itself

    >>> number_or_symbol('8')
    8
    >>> number_or_symbol('-5.32')
    -5.32
    >>> number_or_symbol('1.2.3.4')
    '1.2.3.4'
    >>> number_or_symbol('x')
    'x'
    """
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return x


def tokenize(source):
    """
    Splits an input string into meaningful tokens (left parens, right parens,
    other whitespace-separated values).  Returns a list of strings.

    Arguments:
        source (str): a string containing the source code of a Scheme
                      expression
    """
    tokenized_list = []
    # var to see if char are included in comment i.e ; abcd
    is_comment = False

    expression = []
    
    #Iterate through source and check for '(', ')', ; and everything else
    for char in source:
        #Check if character is part of a comment, if so then skip
        if is_comment:
            #If the char is a new line then it's no longer comment
            if char == '\n':
                is_comment = False
        else:
            if char == '(':
                tokenized_list.append('(')
            elif char == ')':
                # if expression not empty then it was multi-character and add it to tokenized_list
                if expression:
                    #Combine the characters into one string
                    expression = ''.join(expression)
                    #Add it
                    tokenized_list.append(expression)
                    #Reset back to empty expression
                    expression = []
                #add ')' after it
                tokenized_list.append(')')
            #if character is a whitespace or newline character
            elif char == ' ' or char == '\n':
                #If expression is multi-character join it and add to tokenized_list
                if expression:
                    expression = ''.join(expression)
                    tokenized_list.append(expression)
                    #Reset expression
                    expression = []
            # If we hit a comment set is_comment to true and ignore each iteration until not comment
            elif char == ';':
                is_comment = True
            # if none of the above then it must be everything else
            else:
                expression.append(char)
    # if there's one last multi-line expression add it
    if expression:
        expression = ''.join(expression)
        tokenized_list.append(expression)
    
    return tokenized_list

#Checks if tokens give a valid expression for them to be parsed
def is_valid_tokens(tokens):
    not_paired = 0
    parenthesis_seen = False
    for token in tokens:
        if token == '(':
            parenthesis_seen = True
            if not_paired < 0:
                raise SchemeSyntaxError()
            else:
                not_paired += 1
        elif token == ')':
            not_paired -= 1
    if not_paired != 0:
        raise SchemeSyntaxError()
    if not parenthesis_seen and len(tokens) > 1:
        raise SchemeSyntaxError()

def parse(tokens):
    """
    Parses a list of tokens, constructing a representation where:
        * symbols are represented as Python strings
        * numbers are represented as Python ints or floats
        * S-expressions are represented as Python lists

    Arguments:
        tokens (list): a list of strings representing tokens
    """
    def parse_expression(index):
        if index == len(tokens):
            return None, None
        #Get the token
        token = number_or_symbol(tokens[index])

        #Recursive step
        #Case for if token is a number
        if type(token) is int or type(token) is float:
            return token, index + 1
        #Case for when token is a var
        if type(token) is str and token != '(':
            return token, index + 1
        # S-exp case: token at index i is '('
        S_exp = []
        next_token, next_index = parse_expression(index+1)
        while next_token != ')' and next_index != None:
            S_exp.append(next_token)
            next_token, next_index = parse_expression(next_index)
        # parsed_token, next_index = parse_expression(index+1)
        # return [parsed_token, parse_expression(next_index)], next_index+1
        return S_exp, next_index

    #Check if tokens is a valid expression in before running parse_expression
    #If it's not, it will rais a SchemeSyntaxError()
    is_valid_tokens(tokens)
    result = parse_expression(0)[0]
    return result



######################
# Built-in Functions #
######################

def multiply(args):
    def func(args=args):
        result = 1
        for num in args:
            result *= num
        return result
    return func(args)

def division(args):
    def func(args=args):
        result = args[0]
        for num in args[1:]:
            result /= num
        return result
    return func(args)

scheme_builtins = {
    "+": sum,
    "-": lambda args: -args[0] if len(args) == 1 else (args[0] - sum(args[1:])),
    "*": multiply,
    "/": division,
}


##############
# Evaluation #
##############


def evaluate(tree):
    """
    Evaluate the given syntax tree according to the rules of the Scheme
    language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the
                            parse function
    """
    # If the expression is a list (representing an S-expression), each of the elements in the 
    # list should be evaluated, and the result of evaluating the first element (a function) 
    # should be called with the remaining elements passed in as arguments2. The overall 
    # result of evaluating such a function is the return value of that function call.
    if isinstance(tree, list):
        #Evaluate the function
        first_element = evaluate(tree[0])
        #If first_element is a callable function
        if callable(first_element):
            #Evaluate each other element
            args = [evaluate(arg) for arg in tree[1:]]
            #Pass the evaluated elements into function and return
            return first_element(args)
        #The first element must have not been a valid function
        raise SchemeEvaluationError
        

    #If the expression is a symbol representing a name in scheme_builtins, 
    #it should return the associated object.
    elif tree in scheme_builtins:
        #Return the object itself
        return scheme_builtins[tree]
    
    #If the expression is a number, it should return that number.
    elif isinstance(tree, (int, float)):
        return tree


########
# REPL #
########


def repl(raise_all=False):
    while True:
        # read the input.  pressing ctrl+d exits, as does typing "EXIT" at the
        # prompt.  pressing ctrl+c moves on to the next prompt, ignoring
        # current input
        try:
            inp = input("in> ")
            if inp.strip().lower() == "exit":
                print("  bye bye!")
                return
        except EOFError:
            print()
            print("  bye bye!")
            return
        except KeyboardInterrupt:
            print()
            continue

        try:
            # tokenize and parse the input, then evaluate and print the result
            tokens = tokenize(inp)
            ast = parse(tokens)
            print("  out> ", evaluate(ast))
        except SchemeError as e:
            # if raise_all was given as True, then we want to raise the
            # exception so we see a full traceback.  if not, just print some
            # information about it and move on to the next step.
            #
            # regardless, all Python exceptions will be raised.
            if raise_all:
                raise
            print(f"{e.__class__.__name__}:", *e.args)
        print()


if __name__ == "__main__":
    # code in this block will only be executed if lab.py is the main file being
    # run (not when this module is imported)

    # uncommenting the following line will run doctests from above
    # doctest.testmod()
    #repl()
    #print(scheme_builtins['+']([1, 2, 3]))
    print(evaluate([
    "*",
    7,
    8
  ]))