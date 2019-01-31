"""A module for storing convergence parameters of an optimization and
graphing optimization data.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from quantum_optimal_control.helper_functions.grape_functions import sort_ev
from quantum_optimal_control.core.iterationdatum import IterationDatum

# https://matplotlib.org/faq/usage_faq.html#what-is-a-backend
plt.switch_backend('cairo')

class Convergence:

    def __init__(self, sys_para, time_unit, convergence):
        # paramters
        self.sys_para = sys_para
        self.time_unit = time_unit

        if 'rate' in convergence:
            self.rate = convergence['rate']
        else:
            self.rate = 0.01

        if 'update_step' in convergence:
            self.update_step = convergence['update_step']
        else:
            self.update_step = 100

        if 'conv_target' in convergence:
            self.conv_target = convergence['conv_target']
        else:
            self.conv_target = 1e-8

        if 'max_iterations' in convergence:
            self.max_iterations = convergence['max_iterations']
        else:
            self.max_iterations = 5000

        if 'learning_rate_decay' in convergence:
            self.learning_rate_decay = convergence['learning_rate_decay']
        else:
            self.learning_rate_decay = 2500

        if 'min_grad' in convergence:
            self.min_grad = convergence['min_grad']
        else:
            self.min_grad = 1e-25

        self.iteration_data = []
        self.learning_rate = []
        self.last_iter = 0
        self.accumulate_rate = 1.00
        self.concerned = self.sys_para.states_concerned_list

        if self.sys_para.save_plots:
            plt.figure()

    def store_plot_data(self, loss, reg_loss, anly, run_time):
        """A function to store the data of an iteration on an update step.
        """
        # Take data from this iteration, put it in an IterationData object
        # and store that object in self.iteration_data for later use.
        iteration_num = self.update_step + self.last_iter 
        self.iteration_data.append(IterationDatum(iteration_num, loss, reg_loss, 
                                                  anly, run_time))
        # Update iteration cursor.
        self.last_iter = iteration_num


    def plot_inter_vecs_general(self, pop_inter_vecs, start):
        # plot state evolution
        if self.sys_para.draw_list != []:
            for kk in range(len(self.sys_para.draw_list)):
                plt.plot(np.array([self.sys_para.dt * ii for ii in range(self.sys_para.steps+1)]), np.array(
                    pop_inter_vecs[self.sys_para.draw_list[kk], :]), label=self.sys_para.draw_names[kk])

        else:

            if start > 4:
                plt.plot(np.array([self.sys_para.dt * ii for ii in range(self.sys_para.steps+1)]),
                         np.array(pop_inter_vecs[start, :]), label='Starting level '+str(start))

            for jj in range(4):

                plt.plot(np.array([self.sys_para.dt * ii for ii in range(self.sys_para.steps+1)]),
                         np.array(pop_inter_vecs[jj, :]), label='level '+str(jj))

        forbidden = np.zeros(self.sys_para.steps+1)
        if 'states_forbidden_list' in self.sys_para.reg_coeffs:
            # summing all population of forbidden states
            for forbid in self.sys_para.reg_coeffs['states_forbidden_list']:
                if self.sys_para.dressed_info is None or ('forbid_dressed' in 
                  self.sys_para.reg_coeffs and self.sys_para.reg_coeffs['forbid_dressed']):
                    forbidden = forbidden + np.array(pop_inter_vecs[forbid, :])
                else:
                    v_sorted = sort_ev(self.sys_para.v_c,
                                       self.sys_para.dressed_id)
                    dressed_vec = np.dot(v_sorted, np.sqrt(pop_inter_vecs))
                    forbidden = forbidden + \
                        np.array(np.square(np.abs(dressed_vec[forbid, :])))

            plt.plot(np.array([self.sys_para.dt * ii for ii in range(self.sys_para.steps+1)]),
                     forbidden, label='forbidden', linestyle='--', linewidth=4)

        plt.ylabel('Population')
        plt.ylim(-0.1, 1.1)
        plt.xlabel('Time (' + self.time_unit+')')
        plt.legend(ncol=7)

    def generate_plots(self):
        """Generate some nice plots for the user."""
        
        # Aggregate data from self.iteration_data.
        iter_nums = np.array([iteration.num for iteration in self.iteration_data]) 
        losses = np.array([iteration.loss for iteration in self.iteration_data])
        reg_losses = np.array([iteration.reg_loss for iteration in self.iteration_data])
        
        # Get data for state plots from the final iteration. A form of the 
        # deprecated 'save_evol' function.
        final_iteration = self.iteration_data[-1]
        anly = final_iteration.analysis
        if not self.sys_para.state_transfer:
            final_state = anly.get_final_state()

        # Unkown schuster code.
        i1 = 0
        i2 = 0
        if self.sys_para.state_transfer:
            i2 = i2-1
        gs = gridspec.GridSpec(3+i1+i2+len(self.concerned), 2)
        index = 0

        # Plot error
        # Display loss values for the final iteration.
        loss_plot_title = ("Fidelity Error = {0:1.2f}; All Error = {1:1.2f}; "
            "Unitary Metric: {2:.5f}; Runtime: {3:.1f}").format(final_iteration.loss, 
            final_iteration.reg_loss - final_iteration.loss, 
            anly.tf_unitary_scale.eval(), final_iteration.run_time)
        plt.subplot(gs[index, :], title=loss_plot_title)
        index += 1
        plt.plot(iter_nums, losses, 'bx-', label='Fidelity Error')
        plt.plot(iter_nums, reg_losses, 'go-', label='All Penalties')
        plt.ylabel('Error')
        plt.xlabel('Iteration')
        try:
            plt.yscale('log')
        except:
            plt.yscale('linear')

        plt.legend() 

        # Plot unitary evolution
        if not self.sys_para.state_transfer:
            M = final_state
            plt.subplot(gs[index, 0], title="operator: real")
            plt.imshow(M.real, interpolation='none')
            plt.clim(-1, 1)
            plt.colorbar()
            plt.subplot(gs[index, 1], title="operator: imaginary")
            plt.imshow(M.imag, interpolation='none')
            plt.clim(-1, 1)
            plt.colorbar()
            index += 1

        # Plot operators
        plt.subplot(gs[index, :], title="Simulation Weights")
        ops_weight = anly.get_ops_weight()

        for jj in range(self.sys_para.ops_len):

            plt.plot(np.array([self.sys_para.dt * ii for ii in range(self.sys_para.steps)]), np.array(
                self.sys_para.ops_max_amp[jj]*ops_weight[jj, :]), label='u'+self.sys_para.Hnames[jj])

        # Plot control fields
        plt.title('Optimized pulse')
        plt.ylabel('Amplitude')
        plt.xlabel('Time (' + self.time_unit+')')
        plt.legend()

        index += 1

        # Plot state evolution
        if self.sys_para.use_inter_vecs:
            inter_vecs = anly.get_inter_vecs()
            for ii in range(len(self.concerned)):
                plt.subplot(gs[index+ii, :], title="Evolution")

                pop_inter_vecs = inter_vecs[ii]
                self.plot_inter_vecs_general(
                    pop_inter_vecs, self.concerned[ii])

        fig = plt.gcf()
        if self.sys_para.state_transfer:
            plots = 2
        else:
            plots = 3

        fig.set_size_inches(15, int(plots+len(self.concerned)*18))

    def write_plots(self):
        """Write the current plots to a file."""
        # Notify the user that we're waiting on plots.
        print("Writing plots...")
        self.generate_plots()
        file_name = self.sys_para.file_path.replace(".h5", "_plots.png")
        plt.savefig(file_name, format="png")
