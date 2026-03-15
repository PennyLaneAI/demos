import pennylane as qml
qml.capture.enable()
from pennylane import numpy as np

def non_identity_obs(obs):
    return [o for o in obs if not isinstance(o, qml.Identity)]

class PerturbativeGadgets:
    """ Class to generate the gadget Hamiltonian corresponding to a given
    computational hamiltonian according to the gadget construction derived
    by Faehrmann & Cichy 
    
    Args: 
        perturbation_factor (float) : parameter controlling the magnitude of the
                                      perturbation (aa pre-factor to \lambda_max)
    """
    def __init__(self, perturbation_factor=1):
        self.perturbation_factor = perturbation_factor
    
    def gadgetize(self, Hamiltonian, target_locality=3):
        """Generation of the perturbative gadget equivalent of the given 
        Hamiltonian according to the proceedure in Cichy, Fährmann et al.
        Args:
            Hamiltonian (qml.Hamiltonian)   : target Hamiltonian to decompose
                                              into more local terms
            target_locality (int > 2)       : desired locality of the resulting 
                                              gadget Hamiltonian
        Returns:
            Hgad (qml.Hamiltonian)          : gadget Hamiltonian
        """
        # checking for unaccounted for situations
        self.run_checks(Hamiltonian, target_locality)
        computational_qubits, computational_locality, computational_terms = self.get_params(Hamiltonian)
        Hamiltonian_coeffs, Hamiltonian_ops = Hamiltonian.terms()
        
        # total qubit count, updated progressively when adding ancillaries
        total_qubits = computational_qubits
        #TODO: check proper convergence guarantee
        gap = 1
        perturbation_norm = np.sum(np.abs(Hamiltonian_coeffs)) \
                          + computational_terms * (computational_locality - 1)
        lambda_max = gap / (4 * perturbation_norm)
        l = self.perturbation_factor * lambda_max
        sign_correction = (-1)**(computational_locality % 2 + 1)
        # creating the gadget Hamiltonian
        coeffs_anc = []
        coeffs_pert = []
        obs_anc = []
        obs_pert = []
        ancillary_register_size = int(computational_locality / (target_locality - 2))
        for str_count, string in enumerate(Hamiltonian_ops):
            previous_total = total_qubits
            total_qubits += ancillary_register_size
            # Generating the ancillary part
            for anc_q in range(previous_total, total_qubits):
                coeffs_anc += [0.5, -0.5]
                obs_anc += [qml.Identity(anc_q), qml.PauliZ(anc_q)]
            # Generating the perturbative part
            for anc_q in range(ancillary_register_size):
                term = qml.PauliX(previous_total+anc_q) @ qml.PauliX(previous_total+(anc_q+1)%ancillary_register_size)
                term = qml.prod(term, *non_identity_obs(string.operands)[
                    (target_locality-2)*anc_q:(target_locality-2)*(anc_q+1)])
                obs_pert.append(term)
            coeffs_pert += [l * sign_correction * Hamiltonian_coeffs[str_count]] \
                        + [l] * (ancillary_register_size - 1)
        coeffs = coeffs_anc + coeffs_pert
        obs = obs_anc + obs_pert
        Hgad = qml.Hamiltonian(coeffs, obs)
        return Hgad

    def get_params(self, Hamiltonian):
        """ retrieving the parameters n, k and r from the given Hamiltonian
        Args:
            Hamiltonian (qml.Hamiltonian) : Hamiltonian from which to get the
                                            relevant parameters
        Returns:
            computational_qubits (int)    : total number of qubits acted upon by 
                                            the Hamiltonian
            computational_locality (int)  : maximum number of qubits acted upon
                                            by a single term of the Hamiltonian
            computational_terms (int)     : number of terms in the sum 
                                            composing the Hamiltonian
        """
        _, Hamiltonian_ops = Hamiltonian.terms()
        # checking how many qubits the Hamiltonian acts on
        computational_qubits = len(Hamiltonian.wires)
        # getting the number of terms in the Hamiltonian
        computational_terms = len(Hamiltonian_ops)
        # getting the locality, assuming all terms have the same
        computational_locality = max([len(non_identity_obs(Hamiltonian_ops[s])) 
                                      for s in range(computational_terms)])
        return computational_qubits, computational_locality, computational_terms
    
    def run_checks(self, Hamiltonian, target_locality):
        """ method to check a few conditions for the correct application of 
        the methods
        Args:
            Hamiltonian (qml.Hamiltonian) : Hamiltonian of interest
            target_locality (int > 2)     : desired locality of the resulting 
                                            gadget Hamiltonian
        Returns:
            None
        """
        _, Hamiltonian_ops = Hamiltonian.terms()
        computational_qubits, computational_locality, _ = self.get_params(Hamiltonian)
        computational_qubits = len(Hamiltonian.wires)
        if computational_qubits != Hamiltonian.wires[-1] + 1:
            raise Exception('The studied computational Hamiltonian is not acting on ' + 
                            'the first {} qubits. '.format(computational_qubits) + 
                            'Decomposition not implemented for this case')
        # Check for same string lengths
        localities=[]
        for string in Hamiltonian_ops:
            localities.append(len(non_identity_obs(string)))
        if len(np.unique(localities)) > 1:
            raise Exception('The given Hamiltonian has terms with different locality.' +
                            ' Gadgetization not implemented for this case')
        # validity of the target locality given the computational locality
        if target_locality < 3:
            raise Exception('The target locality can not be smaller than 3')
        ancillary_register_size = computational_locality / (target_locality - 2)
        if int(ancillary_register_size) != ancillary_register_size:
            raise Exception('The locality of the Hamiltonian and the target' + 
                             ' locality are not compatible. The gadgetization' + 
                             ' with "unfull" ancillary registers is not' + 
                             ' supported yet. Please choose such that the' + 
                             ' computational locality is divisible by the' + 
                             ' target locality - 2')



