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

#Class fo defining functions
class Function():
    #initialize the body of function, parameters, and frame defined in
    def __init__(self, parameters, body, frame):
        self.parameters = parameters
        self.body = body
        self.frame = frame

    def __call__(self, args):
        #Check to see if enough arguments were provided
        if len(self.parameters) != len(args):
            raise SchemeEvaluationError
        #make a new frame whose parent is the function's enclosing frame (this is called lexical scoping).
        frame = Frame(self.frame)
        #Assign each arg in args to each parameter
        for i, parameter in enumerate(self.parameters):
            frame.bindings[parameter] = args[i]
        return evaluate(self.body, frame)

#Class for frames
class Frame():
    def __init__(self, parent=None):
        #Point to parent frame
        self.parent = parent
        #Create dictionary for variable bindings
        self.bindings = {}

    #We will need to be able to look up variable names in the frame
    def lookup_var(self, var_name):
        #If variable exists within this frames bindings, return it
        if var_name in self.bindings:
            return self.bindings[var_name]
        # Check if frame has a parent if not, raise error
        elif self.parent == None:
            raise SchemeNameError
        #Recursively check the next parent frame for the variable
        else:
            return self.parent.lookup_var(var_name)

    def set_var(self, var_name, expression):
        #If variable exists within this frames bindings, bind it to val
        if var_name in self.bindings:
            #Assign it
            self.bindings[var_name] = expression
            #When defined return the val of the exp
            return expression
        # Check if frame has a parent if not, raise error
        elif self.parent == None:
            raise SchemeNameError
        #Recursively check the next parent frame for the variable
        else:
            return self.parent.set_var(var_name, expression)

    #Delete variable
    def delete_var(self, var):
        if var in self.bindings:
            return self.bindings.pop(var)
        raise SchemeNameError

class Pair():
    def __init__(self, car, cdr):
        self.car = car
        self.cdr = cdr
        
    def get_car(self):
        return self.car

    def get_cdr(self):
        return self.cdr

    def set_car(self, val):
        self.car = val

    def set_cdr(self, val):
        self.cdr = val


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

def result_and_frame(tree, frame=None):
    #Check if frame is None then this must be the global frame and its parent should be built-ins
    if frame == None:
        parent = Frame()
        parent.bindings = scheme_builtins
        frame = Frame(parent)
    return evaluate(tree, frame), frame

#should evaluate to true if all of its arguments are equal to each other.
def equal(args):
    if [args[0]] * len(args) == args:
        return True
    return False

#should evaluate to true if its arguments are in decreasing order. (>)
def greater_than(args):
    return all(s > t for s, t in zip(args, args[1:]))

#should evaluate to true if its arguments are in nonincreasing order.
def greater_than_or_equal_to(args):
    l = args[:]
    l.sort(reverse=True)
    if l == args:
        return True
    return False

#should evaluate to true if its arguments are in increasing order.
def less_than(args):
    return all(s < t for s, t in zip(args, args[1:]))

#should evaluate to true if its arguments are in nondecreasing order.
def less_than_or_equal_to(args):
    l = args[:]
    l.sort()
    if l == args:
        return True
    return False

#should be a special form that takes arbitrarily many arguments and evaluates to true if all of its arguments are true.
def _and(args):
    for arg in args:
        if not evaluate(arg):
            return False
    return True

#should be a special form that takes arbitrarily many arguments and evaluates to true if any of its arguments is true.
def _or(args):
    for arg in args:
        if evaluate(arg):
            return True
    return False

#should be a built-in function that takes a single argument and should evaluate to false if its argument is true and true if its argument is false.
def _not(arg):
    if len(arg) == 1:
        return not arg[0]
    raise SchemeEvaluationError

def cons(args):
    #check for correct number of arguments
    if len(args) != 2:
        raise SchemeEvaluationError
    else:
        return Pair(*args)

def car(arg):
    if not isinstance(arg, Pair):
        raise SchemeEvaluationError
    return arg.get_car()

def cdr(arg):
    if not isinstance(arg, Pair):
        raise SchemeEvaluationError
    return arg.get_cdr()

#should take an arbitrary object as input, and it should return #t if that object is a linked list, and #f otherwise.
def is_list(args):
    if isinstance(args, list):
        return is_list(args[0])
    elif isinstance(args, Pair):
        return is_list(args.get_cdr())
    elif args == None:
        return True
    return False

def len_helper(args):
    #Start at the beginning of list
    temp = args[0]
    count = 0
    #Iterate through list until we get to the end and return count
    while temp:
        count += 1
        temp = temp.get_cdr()
    return count

#should take a list as argument and should return the length of that list. When called on any object that is not a linked list, it should raise a SchemeEvaluationError.
def list_len(args):
    # if empty list, return 0
    if args[0] == None:
        return 0
    # if not a linked list
    if not is_list([args[0]]):
        raise SchemeEvaluationError
    return len_helper(args)

def index(args):
    """
    Returns the item at index i given a linked list
    """
    # separate args into list and index
    list, idx = args
    #Get the head car
    curr = list
    #current index
    curr_idx = 0
    #Iterate through until wanted index is reached
    while isinstance(curr, Pair):
        if curr_idx == idx:
            return curr.get_car()
        curr_idx += 1
        curr = curr.get_cdr()
    #If we get here then error
    raise SchemeEvaluationError

#create new copy of list
def get_copy_list(args):
    if args == None:
        return None
    # if not a list rais an error
    elif not isinstance(args, Pair):
        raise SchemeEvaluationError
    # recursively create copy
    elif args.get_cdr() == None:
        return Pair(args.get_car(), None)
    return Pair(args.get_car(), get_copy_list(args.get_cdr()))

def append(args):
    #Create copy of list
    copy = []
    for element in args:
        if element != None:
            copy.append(get_copy_list(element))
    #Check for empty
    if copy == []:
        return None
    #Just return copy if len 1
    elif len(copy) == 1:
        return Pair(copy[0].get_car(), copy[0].get_cdr()) 
    #Iterate through and add lists together
    curr = copy[0]
    car = curr
    for i in range(len(copy)-1):
        while curr.get_cdr() != None:    
            curr = curr.get_cdr()
        curr.set_cdr(copy[i+1])
    return car

def make_list(args):
    if len(args) == 0:
        return None
    else:
        return Pair(args[0], make_list(args[1:]))

def return_car(args):
    if len(args) == 1 and isinstance(args[0], Pair):
        return args[0].get_car()
    raise SchemeEvaluationError

def return_cdr(args):
    if len(args) == 1 and isinstance(args[0], Pair):
        return args[0].get_cdr()
    raise SchemeEvaluationError

def begin(args):
    #Return last element
    return args[-1]

def evaluate_file(file_name, frame=None):
    new_f = ""
    f = open(file_name, 'r')
    line = f.read().splitlines()
    for exp in line:
        new_f = new_f + exp + " "
    return result_and_frame(parse(tokenize(new_f)), frame)[0]

scheme_builtins = {
    "+": sum,
    "-": lambda args: -args[0] if len(args) == 1 else (args[0] - sum(args[1:])),
    "*": multiply,
    "/": division,
    "#t": True,
    "#f": False,
    "equal?": equal,
    ">": greater_than,
    ">=": greater_than_or_equal_to,
    "<": less_than,
    "<=": less_than_or_equal_to,
    "and": _and,
    "or": _or,
    "not": _not,
    "cons": cons,
    "list?": is_list,
    "length": list_len,
    "list-ref": index,
    "nil": None,
    "append": append,
    "list": make_list,
    "car": return_car,
    "cdr": return_cdr,
    "begin": begin,
}


##############
# Evaluation #
##############


def evaluate(tree, frame=None):
    """
    Evaluate the given syntax tree according to the rules of the Scheme
    language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the
                            parse function
    """
    #Check if frame is None then this must be the global frame and its parent should be built-ins
    if frame == None:
        parent = Frame()
        parent.bindings = scheme_builtins
        frame = Frame(parent)

    #If the expression is a number, it should return that number.
    if isinstance(tree, (int, float)):
        return tree

    #If expression is a variable return it
    elif isinstance(tree, str):
        return frame.lookup_var(tree)

    elif not isinstance(tree, (int, float)):
        #Empty
        if tree == None:
            return None
        elif tree == []:
            raise SchemeEvaluationError
        #Behavior for defining variables
        if tree[0] == 'define':
            #If second item is still a list its a func def
            if isinstance(tree[1], list):
                #Get func header (name and args)
                header = tree[1]
                if len(header) > 1:
                    #Get the arguments in func header
                    args = header[1:]
                else:
                    args = []
                func = ['lambda', args, tree[2]]
                val = evaluate(func, frame)
                frame.bindings[tree[1][0]] = val
                return val
            name, expression = tree[1], tree[2]
            val = evaluate(expression, frame)
            frame.bindings[name] = val
            return val
        elif tree[0] == 'del':
            return frame.delete_var(tree[1])
        #Behavior for nil/empty list
        elif tree[0] == 'nil':
            return None
        #Behavior for if statments
        elif tree[0] == 'if':
            if evaluate(tree[1], frame):
                return evaluate(tree[2], frame)
            else:
                return evaluate(tree[3], frame)
        #Behavior for defining functions with lambda
        elif tree[0] == 'let':
            new_bindings = {}
            for binding in tree[1]:
                evaluated = evaluate(binding[1], frame)
                new_bindings[binding[0]] = evaluated
            new_frame = Frame(frame)
            new_frame.bindings = new_bindings
            return evaluate(tree[2], new_frame)
        elif tree[0] == 'lambda':
            parameters, body = tree[1], tree[2]
            return Function(parameters, body, frame)
        elif tree[0] == 'set!':
            evaluated = evaluate(tree[2], frame)
            return frame.set_var(tree[1], evaluate(tree[2], frame))
        #Otherwise, e is a compound expression representing a function call
        else:
            #Evaluate the function
            first_element = evaluate(tree[0], frame) 
            #Case for and and or functions
            if tree[0] == 'and' or tree[0] == 'or':
                args = (evaluate(arg, frame) for arg in tree[1:])
                return first_element(args)
            #If first_element is a callable function
            elif callable(first_element):
                #Evaluate each other element
                args = [evaluate(arg, frame) for arg in tree[1:]]
                #Pass the evaluated elements into function and return
                return first_element(args)
            #The first element must have not been a valid function
            raise SchemeEvaluationError


########
# REPL #
########


# def repl(raise_all=False):
#     while True:
#         # read the input.  pressing ctrl+d exits, as does typing "EXIT" at the
#         # prompt.  pressing ctrl+c moves on to the next prompt, ignoring
#         # current input
#         try:
#             inp = input("in> ")
#             if inp.strip().lower() == "exit":
#                 print("  bye bye!")
#                 return
#         except EOFError:
#             print()
#             print("  bye bye!")
#             return
#         except KeyboardInterrupt:
#             print()
#             continue

#         try:
#             # tokenize and parse the input, then evaluate and print the result
#             tokens = tokenize(inp)
#             ast = parse(tokens)
#             print("  out> ", evaluate(ast))
#         except SchemeError as e:
#             # if raise_all was given as True, then we want to raise the
#             # exception so we see a full traceback.  if not, just print some
#             # information about it and move on to the next step.
#             #
#             # regardless, all Python exceptions will be raised.
#             if raise_all:
#                 raise
#             print(f"{e.__class__.__name__}:", *e.args)
#         print()

def repl(raise_all=False):
    global_frame = None
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
            # tokenize and parse the input
            tokens = tokenize(inp)
            ast = parse(tokens)
            # if global_frame has not been set, we want to call
            # result_and_frame without it (which will give us our new frame).
            # if it has been set, though, we want to provide that value
            # explicitly.
            args = [ast]
            if global_frame is not None:
                args.append(global_frame)
            result, global_frame = result_and_frame(*args)
            # finally, print the result
            print("  out> ", result)
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
    repl()
    # s = tokenize('(append (list 9 8 7))')
    # z = parse(s)
    # print(evaluate(z))