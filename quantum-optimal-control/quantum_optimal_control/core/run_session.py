import time

import numpy as np
import tensorflow as tf
from .analysis import Analysis
from scipy.optimize import minimize
from quantum_optimal_control.helper_functions.data_management import H5File


class run_session:
    def __init__(self, tfs, graph, conv, sys_para, method, single_simulation=False, use_gpu=True):
        self.tfs = tfs
        self.graph = graph
        self.conv = conv
        self.sys_para = sys_para
        self.update_step = conv.update_step
        self.iterations = 0
        self.method = method.upper()
        self.target = False
        self.end = False
        self.conv_time = 0.
        self.conv_iter = 0
        self.elapsed = 0.
        self.anly = None
        if not use_gpu:
            config = tf.ConfigProto(device_count={'GPU': 0})
        else:
            config = None

        with tf.Session(graph=graph, config=config) as self.session:

            tf.global_variables_initializer().run()

            print("Initialized")

            if self.method == 'EVOLVE':
                self.start_time = time.time()
                x0 = self.sys_para.ops_weight_base
                self.l, self.rl, self.grads, self.metric, self.g_squared = self.get_error(
                    x0)
                self.get_end_results()

            else:
                if self.method != 'ADAM':  # Any BFGS scheme
                    self.bfgs_optimize(method=self.method)

                if self.method == 'ADAM':
                    self.start_adam_optimizer()

    def start_adam_optimizer(self):
        # adam optimizer
        self.start_time = time.time()
        self.end = False
        while True:

            self.g_squared, self.l, self.rl, self.metric = self.session.run(
                [self.tfs.grad_squared, self.tfs.loss, self.tfs.reg_loss, self.tfs.unitary_scale])

            if (self.l < self.conv.conv_target) or (self.g_squared < self.conv.min_grad) \
                    or (self.iterations >= self.conv.max_iterations):
                self.end = True

            self.update_and_save()

            if self.end:
                self.get_end_results()
                break

            learning_rate = float(
                self.conv.rate) * np.exp(-float(self.iterations) / self.conv.learning_rate_decay)
            self.feed_dict = {self.tfs.learning_rate: learning_rate}

            _ = self.session.run([self.tfs.optimizer],
                                 feed_dict=self.feed_dict)

    def update_and_save(self):
        if not self.end:
            # Update and save based on update_step
            update_step_reached = self.iterations % self.conv.update_step == 0            
            if update_step_reached:
                self.anly = Analysis(self.sys_para, self.tfs.final_state, 
                                     self.tfs.ops_weight, self.tfs.unitary_scale,
                                     self.tfs.inter_vecs)
                self.save_data()
                self.display()
                if self.sys_para.save_plots:
                    run_time = time.time() - self.start_time
                    self.conv.store_plot_data(self.l, self.rl, self.anly, run_time)

            self.iterations += 1

    def get_end_results(self):
        # get optimized pulse and propagation
        # get and save inter vects

        self.anly = Analysis(self.sys_para, self.tfs.final_state, self.tfs.ops_weight, 
                             self.tfs.unitary_scale, self.tfs.inter_vecs)
        self.save_data()
        self.uks = self.Get_uks()

        if self.sys_para.save_plots:
            self.conv.write_plots()

        if not self.sys_para.state_transfer:
            self.Uf = self.anly.get_final_state()
        else:
            self.Uf = []

    def Get_uks(self):
        # to get the pulse amplitudes
        uks = self.anly.get_ops_weight()
        for ii in range(len(uks)):
            uks[ii] = self.sys_para.ops_max_amp[ii]*uks[ii]
        return uks

    def get_error(self, uks):
        # get error and gradient for scipy bfgs:
        self.session.run(self.tfs.ops_weight_base.assign(uks))

        g, l, rl, metric, g_squared = self.session.run(
            [self.tfs.grad_pack, self.tfs.loss, self.tfs.reg_loss, self.tfs.unitary_scale, self.tfs.grad_squared])

        final_g = np.transpose(np.reshape(
            g, (len(self.sys_para.ops_c)*self.sys_para.steps)))

        return l, rl, final_g, metric, g_squared

    def save_data(self):
        """Write data to the output file."""
        if self.sys_para.save:
            self.elapsed = time.time() - self.start_time
            with H5File(self.sys_para.file_path) as hf:
                hf.append('error', np.array(self.l))
                hf.append('reg_error', np.array(self.rl))
                hf.append('uks', np.array(self.Get_uks()))
                hf.append('iteration', np.array(self.iterations))
                hf.append('run_time', np.array(self.elapsed))
                hf.append('unitary_scale', np.array(self.metric))

    def display(self):
        """Output current optimization information"""
        print('{:>5}|{:>14.2e}|{:>11.2e}|{:>14.5f}|{:>10.1f}s|{:>21.1f}h'
              .format(self.iterations, self.l, self.rl, self.metric, self.elapsed, -1.))

    def minimize_opt_fun(self, x):
        """Minimization function called by scipy in each iteration."""
        self.l, self.rl, self.grads, self.metric, self.g_squared = self.get_error(
            np.reshape(x, (len(self.sys_para.ops_c), int(len(x)/len(self.sys_para.ops_c)))))
        
        # Check if optimization is satisfactory and should terminate.
        if self.l < self.conv.conv_target:
            self.conv_time = time.time()-self.start_time
            self.conv_iter = self.iterations
            # this update_and_save should likely be called here
            # there was an extra update_and_save originally placed in the function
            # 'get_end_results' that i think this line should account for.
            self.update_and_save()
            self.end = True
            print('Target fidelity reached')
            # set zero grads to terminate the scipy optimization
            self.grads = 0*self.grads  

        self.update_and_save()

        if self.method == 'L-BFGS-B':
            return np.float64(self.rl), np.float64(np.transpose(self.grads))
        else:
            return self.rl, np.reshape(np.transpose(self.grads), [len(np.transpose(self.grads))])

    def bfgs_optimize(self, method='L-BFGS-B', jac=True, options=None):
        """Employ scipy for optimization."""
        self.first = True
        self.conv_time = 0.
        self.conv_iter = 0
        self.end = False

        # Display optimization header and continuous output table.
        print("Starting " + self.method + " optimization.")
        print('{:^5}|{:^14}|{:^11}|{:^14}|{:^11}|{:^21}'
              .format('Iters', 'Fidelity Error', 'All Error', 'Unitary Metric', 
                      'Runtime', 'Est. Remaining Runtime'))

        # Run optimization.
        x0 = self.sys_para.ops_weight_base
        options = {'maxfun': self.conv.max_iterations,
                   'gtol': self.conv.min_grad, 'disp': False, 'maxls': 40}
        self.start_time = time.time()
        res = minimize(self.minimize_opt_fun, x0, method=method, jac=jac, 
                       options=options)
        run_time = time.time() - self.start_time

        # Display optimization summary.
        loss = self.session.run(self.tfs.loss)
        print(self.method + ' optimization complete.')
        print(res.message)
        print("Error: {0:1.2f}".format(loss))
        print("Run time: {0}".format(str(run_time)))
        
        self.get_end_results()
