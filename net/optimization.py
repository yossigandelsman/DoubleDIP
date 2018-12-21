import torch


def optimize(optimizer_type, parameters, optimization_closure,
             plot_closure,
             learning_rate,
             num_iter,
             optimization_closure_args,
             plot_closure_args):
    """
    Runs optimization loop.

    :param optimizer_type: 'LBFGS' of 'adam'
    :param parameters: list of Tensors to optimize over
    :param optimization_closure: function, that returns loss variable
    :param plot_closure: function that plots the loss and other information
    :param learning_rate: learning rate
    :param num_iter: number of iterations
    :param dict optimization_closure_args: the arguments for the optimization closure
    :param dict plot_closure_args: the arguments for the plot closure
    :return:
    """
    if optimizer_type == 'LBFGS':
        assert False

    elif optimizer_type == 'adam':
        print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)

        for j in range(num_iter):
            optimizer.zero_grad()
            optimization_results = optimization_closure(j, **optimization_closure_args)
            if plot_closure:
                plot_closure(j, *optimization_results, **plot_closure_args)
            optimizer.step()
    else:
        assert False


def uneven_optimize(optimizer_type, parameters, optimization_closure, 
                    plot_closure,
                    learning_rate,
                    num_iter, step,
                    optimization_closure_args,
                    plot_closure_args):
    """
    Runs optimization loop.

    :param optimizer_type: 'LBFGS' of 'adam'
    :param parameters: list of Tensors to optimize over
    :param optimization_closure: function, that returns loss variable
    :param plot_closure: function that plots the loss and other information
    :param learning_rate: learning rate
    :param num_iter: number of iterations
    :param dict optimization_closure_args: the arguments for the optimization closure
    :param dict plot_closure_args: the arguments for the plot closure
    :return:
    """
    if optimizer_type == 'LBFGS':
        assert False

    elif optimizer_type == 'adam':
        print('Starting optimization with ADAM')
        next_step_optimization_args = None
        for j in range(num_iter // step):
            optimizer = torch.optim.Adam(parameters, lr=learning_rate)
            for i in range(step):
                optimizer.zero_grad()
                optimization_results, next_step_optimization_args_temp = \
                    optimization_closure(j*step + i, next_step_optimization_args, **optimization_closure_args)
                if plot_closure:
                    plot_closure(j*step + i, *optimization_results, **plot_closure_args)
                optimizer.step()
                if next_step_optimization_args is None:
                    # step zero
                    next_step_optimization_args = next_step_optimization_args_temp
            next_step_optimization_args = next_step_optimization_args_temp
    else:
        assert False


