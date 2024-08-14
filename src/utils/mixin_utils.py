import inspect
import click
def get_nested_signature(nested_cls):
    #this function is applied to a class that is the results of applying mixins to a base class
    # there can be intermediate classes that are the result of applying some mixins to the base class
    # we want to find out which parameters are passed to the base class and the respective mixins
    # we war not only interested in the names but qalso the types and default values if any

    # we will use the __mro__ attribute of the class to get the order in which the classes are inherited
    # we will then iterate over the classes in the order of inheritance and get the signature of the __init__ method
    # of each class
    # we will then extract the parameters from the signature and store them in a dictionary
    # we will then return the dictionary

    # get the order of inheritance
    mro = nested_cls.__mro__
    #initialize a dictionary to store the parameters
    parameters = {}
    # iterate over the classes in the order of inheritance
    for cls in mro:
        # get the signature of the __init__ method
        sig = inspect.signature(cls.__init__)
        # iterate over the parameters
        for param in sig.parameters.values():
            # if the parameter is not self, args or kwargs
            if param.name not in {'self','args','kwargs'}:
                # store the parameter in the dictionary
                parameters[param.name] = param

    parameters_per_class = {}

    for cls in mro:
        sig = inspect.signature(cls.__init__)
        #only inccude cls and parameters that are not self,args,kwargs
        valid_params = {k:v for k,v in sig.parameters.items() if k not in {'self','args','kwargs'}}
        if not valid_params:
            continue
        parameters_per_class[cls.__name__] = valid_params

    return parameters,parameters_per_class


@click.command()
@click.option("--cls_path", help="The path to the class whose signature you want to get")
def get_nested_signature_by_class_path(cls_path:str):
    cls_name = cls_path.split(".")[-1]
    module_path = ".".join(cls_path.split(".")[:-1])
    module = __import__(module_path,fromlist=[cls_name])
    cls = getattr(module,cls_name)
    parameters,parameters_per_class = get_nested_signature(cls)
    print(parameters)
    print(parameters_per_class)




if __name__ == '__main__':
    get_nested_signature_by_class_path()