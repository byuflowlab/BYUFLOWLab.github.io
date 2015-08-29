def uncon(func, x0, epsilon_g, max_iter):
    """An algorithm for unconstrained optimization.

    Parameters
    ----------
    func : function handle
        function handle to a function of the form: f, g = func(x)
        where f is the function value and g is a numpy array containing
        the gradient. x are design variables only.
    x0 : ndarray
        starting point
    epsilon_g : float
        convergence tolerance.  you should terminate when
        np.max(np.abs(g)) <= epsilon_g.  (the infinity norm of the gradient)
    max_iter : int
        the maximum number of function calls.  You should also
        terminate if your algorithm exceeds the maximum number of
        function calls.  I will use a very large number here,
        it is just included so that your code doesn't run forever
        on my machine.

    Outputs
    -------
    xopt : ndarray
        the optimal solution
    fopt : float
        the corresponding function value
    """


    # Your code goes here!  You can (and should) call other functions, but make
    # sure you do not change the function signature for this file.  This is the
    # file I will call to test your algorithm.


    return xopt, fopt