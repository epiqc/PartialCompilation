"""Pass for cancellation of consecutive H gates.
"""
from qiskit.transpiler._basepasses import TransformationPass


class HCancellation(TransformationPass):
    """Cancel back-to-back 'h' gates in dag."""

    def run(self, dag):
        """
        Run one pass of h cancellation on the circuit
        Args:
            dag (DAGCircuit): the directed acyclic graph to run on.
        Returns:
            DAGCircuit: Transformed DAG.
        """
        h_runs = dag.collect_runs(["h"])
        for h_run in h_runs:
            # Partition the h_run into chunks that are on the same wire.
            partition = list()
            chunk = list()
            for i in range(len(h_run) - 1):
                chunk.append(h_run[i])
                qargs0 = dag.multi_graph.node[h_run[i]]["qargs"]
                qargs1 = dag.multi_graph.node[h_run[i + 1]]["qargs"]
                # If the next gate is on a different wire, add this chunk
                # to the partition and start a new chunk.
                if qargs0 != qargs1:
                    partition.append(chunk)
                    chunk = []
            # Always add the last operator to the chunk. Either it is on
            # a different wire than its predecessor in which case a new chunk
            # was created in the last iteration of the for loop on the
            # predecessor, or the chunk is on the same wire and the chunk on
            # its wire is populated.
            chunk.append(h_run[-1])
            partition.append(chunk)
            # Simplify each chunk in the partition
            for chunk in partition:
                if len(chunk) % 2 == 0:
                    for n in chunk:
                        dag._remove_op_node(n)
                else:
                    for n in chunk[1:]:
                        dag._remove_op_node(n)
        return dag
