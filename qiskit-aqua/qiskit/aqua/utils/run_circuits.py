# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import sys
import logging
import time
import copy
import os
import uuid

import numpy as np
from qiskit import compile as q_compile
from qiskit.providers import BaseBackend, JobStatus, JobError
from qiskit.providers.builtinsimulators.simulatorsjob import SimulatorsJob
from qiskit.providers.ibmq.ibmqjob import IBMQJob

from qiskit.aqua.aqua_error import AquaError
from qiskit.aqua.utils import summarize_circuits
from qiskit.aqua.utils.backend_utils import (is_aer_provider,
                                             is_simulator_backend, is_ibmq_provider)

MAX_CIRCUITS_PER_JOB = os.environ.get('QISKIT_AQUA_MAX_CIRCUITS_PER_JOB', None)

logger = logging.getLogger(__name__)


def find_regs_by_name(circuit, name, qreg=True):
    """Find the registers in the circuits.

    Args:
        circuit (QuantumCircuit): the quantum circuit.
        name (str): name of register
        qreg (bool): quantum or classical register

    Returns:
        QuantumRegister or ClassicalRegister or None: if not found, return None.

    """
    found_reg = None
    regs = circuit.qregs if qreg else circuit.cregs
    for reg in regs:
        if reg.name == name:
            found_reg = reg
            break
    return found_reg


def _avoid_empty_circuits(circuits):

    new_circuits = []
    for qc in circuits:
        if len(qc) == 0:
            tmp_q = None
            for q in qc.qregs:
                tmp_q = q
                break
            if tmp_q is None:
                raise NameError("A QASM without any quantum register is invalid.")
            qc.iden(tmp_q[0])
        new_circuits.append(qc)
    return new_circuits


def _reuse_shared_circuits(circuits, backend, backend_config, compile_config, run_config,
                           qjob_config=None, backend_options=None, show_circuit_summary=False):
    """Reuse the circuits with the shared head.

    We assume the 0-th circuit is the shared_circuit, so we execute it first
    and then use it as initial state for simulation.

    Note that all circuits should have the exact the same shared parts.
    """
    qjob_config = qjob_config or {}
    backend_options = backend_options or {}

    shared_circuit = circuits[0]
    shared_result = compile_and_run_circuits(shared_circuit, backend, backend_config,
                                             compile_config, run_config, qjob_config,
                                             show_circuit_summary=show_circuit_summary)

    if len(circuits) == 1:
        return shared_result
    shared_quantum_state = np.asarray(shared_result.get_statevector(shared_circuit))
    # extract different of circuits
    for circuit in circuits[1:]:
        circuit.data = circuit.data[len(shared_circuit):]

    temp_backend_options = copy.deepcopy(backend_options)
    if 'backend_options' not in temp_backend_options:
        temp_backend_options['backend_options'] = {}
    temp_backend_options['backend_options']['initial_statevector'] = shared_quantum_state
    diff_result = compile_and_run_circuits(circuits[1:], backend, backend_config,
                                           compile_config, run_config, qjob_config,
                                           backend_options=temp_backend_options,
                                           show_circuit_summary=show_circuit_summary)
    result = _combine_result_objects([shared_result, diff_result])
    return result


def _combine_result_objects(results):
    """Tempoary helper function.

    TODO:
        This function would be removed after Terra supports job with infinite circuits.
    """
    if len(results) == 1:
        return results[0]

    new_result = copy.deepcopy(results[0])

    for idx in range(1, len(results)):
        new_result.results.extend(results[idx].results)

    return new_result


def compile_and_run_circuits(circuits, backend, backend_config, compile_config, run_config,
                             qjob_config=None, backend_options=None,
                             noise_config=None, show_circuit_summary=False,
                             has_shared_circuits=False, circuit_cache=None,
                             skip_qobj_validation=False, **kwargs):
    """
    An execution wrapper with Qiskit-Terra, with job auto recover capability.

    The autorecovery feature is only applied for non-simulator backend.
    This wraper will try to get the result no matter how long it costs.

    Args:
        circuits (QuantumCircuit or list[QuantumCircuit]): circuits to execute
        backend (BaseBackend): backend instance
        backend_config (dict): configuration for backend
        compile_config (dict): configuration for compilation
        run_config (RunConfig): configuration for running a circuit
        qjob_config (dict): configuration for quantum job object
        backend_options (dict): configuration for simulator
        noise_config (dict): configuration for noise model
        show_circuit_summary (bool): showing the summary of submitted circuits.
        has_shared_circuits (bool): use the 0-th circuits as initial state for other circuits.

    Returns:
        Result: Result object

    Raises:
        AquaError: Any error except for JobError raised by Qiskit Terra
    """
    qjob_config = qjob_config or {}
    backend_options = backend_options or {}
    noise_config = noise_config or {}

    if backend is None or not isinstance(backend, BaseBackend):
        raise ValueError('Backend is missing or not an instance of BaseBackend')

    if not isinstance(circuits, list):
        circuits = [circuits]

    if 'statevector' in backend.name():
        circuits = _avoid_empty_circuits(circuits)

    if has_shared_circuits:
        return _reuse_shared_circuits(circuits, backend, backend_config, compile_config,
                                      run_config, qjob_config, backend_options)

    with_autorecover = False if backend.configuration().simulator else True

    if MAX_CIRCUITS_PER_JOB is not None:
        max_circuits_per_job = int(MAX_CIRCUITS_PER_JOB)
    else:
        if backend.configuration().local:
            max_circuits_per_job = sys.maxsize
        else:
            max_circuits_per_job = backend.configuration().max_experiments

    if circuit_cache is not None and circuit_cache.try_reusing_qobjs:
        # Check if all circuits are the same length. If not, don't try to use the same qobj.experiment for all of them.
        if len(set([len(circ.data) for circ in circuits])) > 1:
            circuit_cache.try_reusing_qobjs = False
        else:  # Try setting up the reusable qobj
            # Compile and cache first circuit if cache is empty. The load method will try to reuse it
            if circuit_cache.try_reusing_qobjs and circuit_cache.qobjs is None:
                qobj = q_compile([circuits[0]], backend, **execute_config)
                circuit_cache.cache_circuit(qobj, [circuits[0]], 0)

    qobjs = []
    jobs = []
    job_ids = []
    chunks = int(np.ceil(len(circuits) / max_circuits_per_job))
    for i in range(chunks):
        sub_circuits = circuits[i * max_circuits_per_job:(i + 1) * max_circuits_per_job]
        if circuit_cache is not None and circuit_cache.misses < circuit_cache.allowed_misses:
            try:
                qobj = circuit_cache.load_qobj_from_cache(sub_circuits, i, run_config=run_config)
            # cache miss, fail gracefully
            except (TypeError, IndexError, FileNotFoundError, EOFError, AquaError, AttributeError) as e:
                circuit_cache.try_reusing_qobjs = False  # Reusing Qobj didn't work
                circuit_cache.clear_cache()
                logger.debug('Circuit cache miss, recompiling. Cache miss reason: ' + repr(e))
                qobj = q_compile(sub_circuits, backend, **backend_config,
                                 **compile_config, **run_config.to_dict())
                circuit_cache.cache_circuit(qobj, sub_circuits, i)
                circuit_cache.misses += 1
        else:
            qobj = q_compile(sub_circuits, backend, **backend_config,
                             **compile_config, **run_config.to_dict())

        if 'expectation' in kwargs:
            from qiskit.providers.aer.utils.qobj_utils import snapshot_instr, append_instr
            # add others, how to derive the correct used number of qubits?
            # the compiled qobj could be wrong if coupling map is used.
            # if mulitple params are provided, we assume that each circuit is corresponding one param
            # otherwise, params are used for all circuits.
            params = kwargs['expectation']['params']
            num_qubits = kwargs['expectation']['num_qubits']
            if len(params) == 1:
                new_ins = snapshot_instr('expectation_value_pauli', 'test', range(num_qubits), params=params[0])
                for ii in range(len(sub_circuits)):
                    qobj = append_instr(qobj, ii, new_ins)
            else:
                for ii in range(len(sub_circuits)):
                    new_ins = snapshot_instr('expectation_value_pauli', 'test', range(num_qubits), params=params[ii])
                    qobj = append_instr(qobj, ii, new_ins)
        # assure get job ids
        while True:
            job = run_on_backend(backend, qobj, backend_options=backend_options, noise_config=noise_config,
                                 skip_qobj_validation=skip_qobj_validation)
            try:
                job_id = job.job_id()
                break
            except JobError as e:
                logger.warning("FAILURE: the {}-th chunk of circuits, can not get job id, "
                               "Resubmit the qobj to get job id. "
                               "Terra job error: {} ".format(i, e))
            except Exception as e:
                logger.warning("FAILURE: the {}-th chunk of circuits, can not get job id, "
                               "Resubmit the qobj to get job id. "
                               "Error: {} ".format(i, e))
        job_ids.append(job_id)
        jobs.append(job)
        qobjs.append(qobj)

    if logger.isEnabledFor(logging.DEBUG) and show_circuit_summary:
        logger.debug(summarize_circuits(circuits))

    results = []
    if with_autorecover:
        logger.info("Backend status: {}".format(backend.status()))
        logger.info("There are {} circuits and they are chunked into {} chunks, "
                    "each with {} circutis (max.).".format(len(circuits), chunks,
                                                           max_circuits_per_job))
        logger.info("All job ids:\n{}".format(job_ids))
        for idx in range(len(jobs)):
            while True:
                job = jobs[idx]
                job_id = job_ids[idx]
                logger.info("Running {}-th chunk circuits, job id: {}".format(idx, job_id))
                # try to get result if possible
                try:
                    result = job.result(**qjob_config)
                    if result.success:
                        results.append(result)
                        logger.info("COMPLETED the {}-th chunk of circuits, "
                                    "job id: {}".format(idx, job_id))
                        break
                    else:
                        logger.warning("FAILURE: the {}-th chunk of circuits, "
                                       "job id: {}".format(idx, job_id))
                except JobError as e:
                    # if terra raise any error, which means something wrong, re-run it
                    logger.warning("FAILURE: the {}-th chunk of circuits, job id: {} "
                                   "Terra job error: {} ".format(idx, job_id, e))
                except Exception as e:
                    raise AquaError("FAILURE: the {}-th chunk of circuits, job id: {} "
                                    "Unknown error: {} ".format(idx, job_id, e)) from e

                # something wrong here, querying the status to check how to handle it.
                # keep qeurying it until getting the status.
                while True:
                    try:
                        job_status = job.status()
                        break
                    except JobError as e:
                        logger.warning("FAILURE: job id: {}, "
                                       "status: 'FAIL_TO_GET_STATUS' "
                                       "Terra job error: {}".format(job_id, e))
                        time.sleep(5)
                    except Exception as e:
                        raise AquaError("FAILURE: job id: {}, "
                                        "status: 'FAIL_TO_GET_STATUS' "
                                        "Unknown error: ({})".format(job_id, e)) from e

                logger.info("Job status: {}".format(job_status))

                # handle the failure job based on job status
                if job_status == JobStatus.DONE:
                    logger.info("Job ({}) is completed anyway, retrieve result "
                                "from backend.".format(job_id))
                    job = backend.retrieve_job(job_id)
                elif job_status == JobStatus.RUNNING or job_status == JobStatus.QUEUED:
                    logger.info("Job ({}) is {}, but encounter an exception, "
                                "recover it from backend.".format(job_id, job_status))
                    job = backend.retrieve_job(job_id)
                else:
                    logger.info("Fail to run Job ({}), resubmit it.".format(job_id))
                    qobj = qobjs[idx]
                    #  assure job get its id
                    while True:
                        job = run_on_backend(backend, qobj,
                                             backend_options=backend_options,
                                             noise_config=noise_config,
                                             skip_qobj_validation=skip_qobj_validation)
                        try:
                            job_id = job.job_id()
                            break
                        except JobError as e:
                            logger.warning("FAILURE: the {}-th chunk of circuits, "
                                           "can not get job id. Resubmit the qobj to get job id. "
                                           "Terra job error: {} ".format(idx, e))
                        except Exception as e:
                            logger.warning("FAILURE: the {}-th chunk of circuits, "
                                           "can not get job id, Resubmit the qobj to get job id. "
                                           "Unknown error: {} ".format(idx, e))
                    jobs[idx] = job
                    job_ids[idx] = job_id
    else:
        results = []
        for job in jobs:
            results.append(job.result(**qjob_config))

    result = _combine_result_objects(results) if len(results) != 0 else None

    return result


# skip_qobj_validation = True does what backend.run and aerjob.submit do, but without qobj validation.
def run_on_backend(backend, qobj, backend_options=None, noise_config=None, skip_qobj_validation=False):
    if skip_qobj_validation:
        job_id = str(uuid.uuid4())
        if is_simulator_backend(backend):
            if is_aer_provider(backend):
                job = backend.run(qobj, **backend_options, **noise_config)
            else:
                job = SimulatorsJob(backend, job_id, backend._run_job, qobj)
                job._future = job._executor.submit(job._fn, job._job_id, job._qobj)
        elif is_ibmq_provider(backend):
            job = IBMQJob(backend, None, backend._api, not is_simulator_backend(backend), qobj=qobj)
            job._future = job._executor.submit(job._fn, job._job_id, job._qobj)
        else:
            logger.info("Can not skip qobj validation for the third-party provider.")
            job = backend.run(qobj, **backend_options, **noise_config)
        return job
    else:
        job = backend.run(qobj, **backend_options, **noise_config)
        return job
