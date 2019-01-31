"""A module to define the PlotData Class."""

class IterationDatum:
    """A class to store the information of a single iteration necessary for 
    quantum optimal control plots.
    
    num (int) - this the mth evaluation of the objective function.
    loss (float?)- the loss.
    reg_loss (float?) - the regular loss.
    analysis (quantum_optimal_control.core.Analysis) - the analysis object.
    run_time (time.time) - the run time.
    """

    def __init__(self, num, loss, reg_loss, analysis, run_time):
        """Store the parameters in the class instance."""
        # 'id' is a built-in function in python, other name than 'num' could be better
        self.num = num 
        self.loss = loss
        self.reg_loss = reg_loss
        self.analysis = analysis
        self.run_time = run_time
        
    
    
