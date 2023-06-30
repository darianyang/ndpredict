"""
Functions for handling command-line input using argparse module.
"""

import argparse
import sys

# import and use gooey conditionally
# adapted from https://github.com/chriskiehl/Gooey/issues/296
try:
    import gooey
    #from gooey import Gooey
    #from gooey import GooeyParser
except ImportError:
    gooey = None

def flex_add_argument(f):
    """Make the add_argument accept (and ignore) the widget option."""

    def f_decorated(*args, **kwargs):
        kwargs.pop('widget', None)
        return f(*args, **kwargs)

    return f_decorated

# Monkey-patching a private class…
argparse._ActionsContainer.add_argument = \
    flex_add_argument(argparse.ArgumentParser.add_argument)

# Do not run GUI if it is not available or if command-line arguments are given.
if gooey is None or len(sys.argv) > 1:
    ArgumentParser = argparse.ArgumentParser

    def gui_decorator(f):
        return f
else:
    ArgumentParser = gooey.GooeyParser
    gui_decorator = gooey.Gooey(
        program_name='NDPredict',
        #navigation='TABBED',
        #advanced=True,
        suppress_gooey_flag=True,
        optional_cols=4, 
        default_size=(1000, 600),
        #tabbed_groups=True,
    )

# TODO: make tabs?
# then push new pip version
@gui_decorator
def create_cmd_arguments(): 
    """
    Use the `argparse` module to make the optional and required command-line
    arguments for the `wedap`. 

    Parameters 
    ----------

    Returns
    -------
    argparse.ArgumentParser: 
        An ArgumentParser that is used to retrieve command line arguments. 
    """
    desc = "================================================= \n" + \
           "=== ASN (N) Deamidation Predictor (NDPredict) === \n" + \
           "================================================= \n" + \
           "\nGiven an input pdb file, calculate N->D deamidation probabilities."

    # create argument parser (gooey based if available)
    if gooey is None:
        parser = argparse.ArgumentParser(description=desc, 
                                        formatter_class=argparse.RawDescriptionHelpFormatter)
    else:
        parser = gooey.GooeyParser(description=desc, 
                                   formatter_class=argparse.RawDescriptionHelpFormatter)

    ##########################################################
    ############### REQUIRED ARGUMENTS #######################
    ##########################################################

    # test out gooey specific widgets
    required = parser.add_argument_group("Required Arguments")
    required.add_argument("-i", "-pdb", "--pdb-input", required=True, nargs="?",
        action="store", dest="pdb", type=str,
        help="The input PDB file.",
        widget="FileChooser")

    ###########################################################
    ############### OPTIONAL ARGUMENTS ########################
    ###########################################################
    # nargs = '?' "One argument will be consumed from the command line if possible, 
        # and produced as a single item. If no command-line argument is present, 
        # the value from default will be produced."

    main = parser.add_argument_group("Main Arguments")
    #optional = parser.add_argument_group("Optional Extra Arguments")

    main.add_argument("-m", "--model", default="ndp_model.pkl", nargs="?",
        dest="model",
        help="NDPredict class object.",
        type=str)  
    main.add_argument("-ci", "--chainid", default="A", nargs="?",
        dest="chainid",
        help="PDB chain specifier.",
        type=str) 
    
    ##########################################################
    ############### FORMATTING ARGUMENTS #####################
    ##########################################################

    formatting = parser.add_argument_group("Plot Formatting Arguments") 

    # formatting.add_argument("--style", default="default", nargs="?",
    #                     dest="style",
    #                     help="mpl style, can leave blank to use default, "
    #                          "input `None` for basic mpl settings, can use a custom "
    #                          "path to a mpl.style text file, or could use a mpl included "
    #                          "named style, e.g. `ggplot`. "
    #                          "Edit the ndpredict/styles/default.mplstyle file to "
    #                          "change default wedap plotting style options.",
    #                     type=str)
    # formatting.add_argument("--cmap", default="viridis", nargs="?",
    #                     dest="cmap", help="mpl colormap name.", type=str)
    formatting.add_argument("--color",
                        dest="color",
                        widget="ColourChooser")
    formatting.add_argument("--xlabel", dest="xlabel", type=str)
    formatting.add_argument("--ylabel", dest="ylabel", type=str)
    #formatting.add_argument("--ylim", help="LB UB", dest="ylim", nargs=2, type=float)
    formatting.add_argument("--title", dest="title", type=str)

    # return the argument parser
    return parser 


# TODO: adjust all to fix str/int/type auto
def handle_command_line(argument_parser): 
    """
    Take command line arguments, check for issues, return the arguments. 

    Parameters
    ----------
    argument_parser : argparse.ArgumentParser 
        The argument parser that is returned in `create_cmd_arguments()`.
    
    Returns
    -------
    argparse.NameSpace
        Contains all arguments passed into wedap.
    
    Raises
    ------  
    Prints specific issues to terminal.
    """
    # retrieve args
    args = argument_parser.parse_args() 

    # h5 file and file exists
    # if not os.path.exists(args.file) or not ".h5" in args.file:  
    #     # print error message and exits
    #     sys.exit("Must input file that exists and is in .h5 file format.")

    # if not args.percentage.isdigit():  # not correct input   
    #     # print out any possible issues
    #     print("You must input a percentage digit. EXAMPLES: \
    #     \n '-p 40' \n You CANNOT add percent sign (eg. 50%) \n You \
    #     CANNOT add decimals (eg. 13.23)") 
    #     sys.exit(0) # exit program

    # # ignore if NoneType since user doesn't want --maxEnsembleSize parameter
    # if args.max_ensemble_size is None: 
    #     pass

    # elif not args.max_ensemble_size.isdigit(): # incorrect input 
    #     # needs to be whole number
    #     print("You must input a whole number with no special characters (eg. 4).")  
    #     sys.exit(0) # exit program 

    # elif args.max_ensemble_size is '0': # user gives 0 to --maxEnsembleSize flag 
    #     print("You cannot input '0' to --maxEnsembleSize flag.")
    #     sys.exit(0) # exit program 

    return args