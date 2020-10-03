import sys

'''
WRITE YOUR CODE BELOW.
'''
from numpy import zeros, float32
#  pgmpy
import pgmpy
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import random
import numpy as np
#You are not allowed to use following set of modules from 'pgmpy' Library.
#
# pgmpy.sampling.*
# pgmpy.factors.*
# pgmpy.estimators.*

def make_power_plant_net():
    """Create a Bayes Net representation of the above power plant problem. 
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    BayesNet = BayesianModel()

    # Add nodes to graph
    BayesNet.add_node("alarm")
    BayesNet.add_node("faulty alarm")
    BayesNet.add_node("gauge")
    BayesNet.add_node("faulty gauge")
    BayesNet.add_node("temperature")

    # Connect nodes
    BayesNet.add_edge("temperature", "gauge")
    BayesNet.add_edge("temperature", "faulty gauge")
    BayesNet.add_edge("faulty gauge", "gauge")
    BayesNet.add_edge("gauge", "alarm")
    BayesNet.add_edge("faulty alarm", "alarm")

    return BayesNet


def set_probability(bayes_net):
    """Set probability distribution for each node in the power plant system.
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    cpd_t = TabularCPD("temperature", 2, values=[[0.8], [0.2]])
    cpd_fa = TabularCPD("faulty alarm", 2, values=[[0.85], [0.15]])
    cpd_fg = TabularCPD("faulty gauge", 2, values=[[0.95, 0.2], [0.05, 0.8]], evidence=["temperature"], evidence_card=[2])
    cpd_g = TabularCPD("gauge", 2, values=[[0.8, 0.05, 0.8, 0.05], [0.2, 0.95, 0.2, 0.95]], evidence=["temperature", "faulty gauge"], evidence_card=[2, 2])
    cpd_a = TabularCPD("alarm", 2, values=[[0.45, 0.1, 0.45, 0.1], [0.55, 0.9, 0.55, 0.9]], evidence=["gauge", "faulty alarm"], evidence_card=[2, 2])
    bayes_net.add_cpds(cpd_t, cpd_fa, cpd_fg, cpd_g, cpd_a)
    return bayes_net


def get_alarm_prob(bayes_net):
    """Calculate the marginal 
    probability of the alarm 
    ringing in the 
    power plant system."""
    solver = VariableElimination(bayes_net)
    alarm_prob = solver.query(variables=["alarm"], joint=False)
    return alarm_prob["alarm"].values[1]


def get_gauge_prob(bayes_net):
    """Calculate the marginal
    probability of the gauge 
    showing hot in the 
    power plant system."""
    solver = VariableElimination(bayes_net)
    gauge_prob = solver.query(variables=["gauge"], joint=False)
    return gauge_prob["gauge"].values[1]


def get_temperature_prob(bayes_net):
    """Calculate the conditional probability 
    of the temperature being hot in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""
    solver = VariableElimination(bayes_net)
    temperature_prob = solver.query(variables=["temperature"], evidence={"alarm": 1, "faulty alarm": 1, "faulty gauge": 1}, joint=False)
    return temperature_prob["temperature"].values[1]


def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """

    # Add nodes
    BayesNet = BayesianModel()
    BayesNet.add_node("A")
    BayesNet.add_node("B")
    BayesNet.add_node("C")
    BayesNet.add_node("AvB")
    BayesNet.add_node("BvC")
    BayesNet.add_node("CvA")

    # Add edges
    BayesNet.add_edge("A", "AvB")
    BayesNet.add_edge("A", "CvA")
    BayesNet.add_edge("B", "AvB")
    BayesNet.add_edge("B", "BvC")
    BayesNet.add_edge("C", "BvC")
    BayesNet.add_edge("C", "CvA")

    # Add probabilities
    cpd_A = TabularCPD("A", 4, values=[[0.15], [0.45], [0.30], [0.10]])
    cpd_B = TabularCPD("B", 4, values=[[0.15], [0.45], [0.30], [0.10]])
    cpd_C = TabularCPD("C", 4, values=[[0.15], [0.45], [0.30], [0.10]])
    cpd_avb = TabularCPD("AvB", 3, values=[
        [0.1, 0.2, 0.15, 0.05, 0.6, 0.1, 0.2, 0.15, 0.75, 0.6, 0.1, 0.2, 0.9, 0.75, 0.6, 0.1],
        [0.1, 0.6, 0.75, 0.9, 0.2, 0.1, 0.6, 0.75, 0.15, 0.2, 0.1, 0.6, 0.05, 0.15, 0.2, 0.1],
        [0.8, 0.2, 0.1, 0.05, 0.2, 0.8, 0.2, 0.1, 0.1, 0.2, 0.8, 0.2, 0.05, 0.1, 0.2, 0.8]
    ], evidence=["A", "B"], evidence_card=[4, 4])
    cpd_bvc = TabularCPD("BvC", 3, values=[
        [0.1, 0.2, 0.15, 0.05, 0.6, 0.1, 0.2, 0.15, 0.75, 0.6, 0.1, 0.2, 0.9, 0.75, 0.6, 0.1],
        [0.1, 0.6, 0.75, 0.9, 0.2, 0.1, 0.6, 0.75, 0.15, 0.2, 0.1, 0.6, 0.05, 0.15, 0.2, 0.1],
        [0.8, 0.2, 0.1, 0.05, 0.2, 0.8, 0.2, 0.1, 0.1, 0.2, 0.8, 0.2, 0.05, 0.1, 0.2, 0.8]
    ], evidence=["B", "C"], evidence_card=[4, 4])
    cpd_avc = TabularCPD("CvA", 3, values=[
        [0.1, 0.2, 0.15, 0.05, 0.6, 0.1, 0.2, 0.15, 0.75, 0.6, 0.1, 0.2, 0.9, 0.75, 0.6, 0.1],
        [0.1, 0.6, 0.75, 0.9, 0.2, 0.1, 0.6, 0.75, 0.15, 0.2, 0.1, 0.6, 0.05, 0.15, 0.2, 0.1],
        [0.8, 0.2, 0.1, 0.05, 0.2, 0.8, 0.2, 0.1, 0.1, 0.2, 0.8, 0.2, 0.05, 0.1, 0.2, 0.8]
    ], evidence=["C", "A"], evidence_card=[4, 4])
    BayesNet.add_cpds(cpd_A, cpd_B, cpd_C, cpd_avb, cpd_bvc, cpd_avc)

    return BayesNet


def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    solver = VariableElimination(bayes_net)
    posterior = solver.query(variables=["BvC"], evidence={"AvB": 0, "CvA": 2}, joint=False)
    return posterior["BvC"].values


def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.    
    """
    if initial_state is None or len(initial_state) == 0:
        initial_state = (
            random.randint(0, 3),
            random.randint(0, 3),
            random.randint(0, 3),
            0,
            random.randint(0, 2),
            2
        )
        return initial_state

    # pick variable to sample
    sample_idx = random.randint(0, 4)
    if sample_idx == 3:
        while sample_idx == 3:
            sample_idx = random.randint(0, 4)

    # Sample the chosen variable
    sample = None
    if sample_idx == 0:
        sample = __calculate_A_posterior__(bayes_net, initial_state)
    if sample_idx == 1:
        sample = __calculate_B_posterior__(bayes_net, initial_state)
    if sample_idx == 2:
        sample = __calculate_C_posterior__(bayes_net, initial_state)
    if sample_idx == 4:
        sample = __calculate_bvc_posterior__(bayes_net, initial_state)
    return sample


def __calculate_bvc_posterior__(bayes_net, initial_state):
    bvc_probs = bayes_net.get_cpds("BvC").values
    b_value = initial_state[1]
    c_value = initial_state[2]

    numerators = []
    for i in range(3):
        numerator = bvc_probs[i][b_value][c_value]
        numerators.append(numerator)

    # normalize numerators
    sum_bvc = sum(numerators)
    likelihoods = np.array(numerators) / sum_bvc
    # Randomly select the new value based on the given distribution
    new_bvc = np.random.choice([0, 1, 2], p=likelihoods)

    return initial_state[0], b_value, c_value, 0, new_bvc, 2


def __calculate_A_posterior__(bayes_net, initial_state):
    # Get relevant values for calculation
    a_probs = bayes_net.get_cpds("A").values
    b_value = initial_state[1]
    c_value = initial_state[2]
    avb_probs = bayes_net.get_cpds("AvB").values[0]
    cva_probs = bayes_net.get_cpds("CvA").values[2]

    likelihood_numerator_a = []
    # Find numerator for posterior calculation
    for i in range(len(a_probs)):
        numerator = a_probs[i] * avb_probs[i][b_value] * cva_probs[c_value][i]
        likelihood_numerator_a.append(numerator)

    # normalize numerators
    sum_a = sum(likelihood_numerator_a)
    likelihoods = np.array(likelihood_numerator_a) / sum_a
    # Randomly select the new value based on the given distribution
    new_a = np.random.choice([0, 1, 2, 3], p=likelihoods)

    return new_a, b_value, c_value, 0, initial_state[4], 2


def __calculate_B_posterior__(bayes_net, initial_state):
    # Get relevant values for calculation
    b_probs = bayes_net.get_cpds("B").values
    a_value = initial_state[0]
    c_value = initial_state[2]
    avb_probs = bayes_net.get_cpds("AvB").values[0]
    bvc_probs = bayes_net.get_cpds("BvC").values[initial_state[4]]

    likelihood_numerator_b = []
    # Find numerator for posterior calculation
    for i in range(len(b_probs)):
        numerator = b_probs[i] * avb_probs[a_value][i] * bvc_probs[i][c_value]
        likelihood_numerator_b.append(numerator)

    # normalize numerators
    sum_b = sum(likelihood_numerator_b)
    likelihoods = np.array(likelihood_numerator_b) / sum_b
    # Randomly select the new value based on the given distribution
    new_b = np.random.choice([0, 1, 2, 3], p=likelihoods)

    return a_value, new_b, c_value, 0, initial_state[4], 2

def __calculate_C_posterior__(bayes_net, initial_state):
    # Get relevant values for calculation
    c_probs = bayes_net.get_cpds("C").values
    b_value = initial_state[1]
    a_value = initial_state[0]
    bvc_probs = bayes_net.get_cpds("AvB").values[initial_state[4]]
    cva_probs = bayes_net.get_cpds("CvA").values[2]

    likelihood_numerator_c = []
    # Find numerator for posterior calculation
    for i in range(len(c_probs)):
        numerator = c_probs[i] * bvc_probs[b_value][i] * cva_probs[i][a_value]
        likelihood_numerator_c.append(numerator)

    # normalize numerators
    sum_c = sum(likelihood_numerator_c)
    likelihoods = np.array(likelihood_numerator_c) / sum_c
    # Randomly select the new value based on the given distribution
    new_c = np.random.choice([0, 1, 2, 3], p=likelihoods)

    return a_value, b_value, new_c, 0, initial_state[4], 2


def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """

    if initial_state is None or len(initial_state) == 0:
        initial_state = (
            random.randint(0, 3),
            random.randint(0, 3),
            random.randint(0, 3),
            0,
            random.randint(0, 2),
            2
        )
        return initial_state

    # Generate random walk
    new_a = __calculate_A_posterior__(bayes_net, initial_state)[0]
    new_b = __calculate_B_posterior__(bayes_net, initial_state)[1]
    new_c = __calculate_C_posterior__(bayes_net, initial_state)[2]
    new_bvc = __calculate_bvc_posterior__(bayes_net, initial_state)[4]
    candidate = (new_a, new_b, new_c, 0, new_bvc, 2)

    # Get cpds
    a_prob = bayes_net.get_cpds("A").values
    b_prob = bayes_net.get_cpds("B").values
    c_prob = bayes_net.get_cpds("C").values
    avb_prob = bayes_net.get_cpds("AvB").values
    bvc_prob = bayes_net.get_cpds("BvC").values
    cva_prob = bayes_net.get_cpds("CvA").values

    # Calculate likelihood of candidate
    p_a = a_prob[new_a]
    p_b = b_prob[new_b]
    p_c = c_prob[new_c]
    p_avb = avb_prob[0][new_a][new_b]
    p_bvc = bvc_prob[new_bvc][new_b][new_c]
    p_cva = cva_prob[2][new_c][new_a]

    p_cand = p_a * p_b * p_c * p_avb * p_bvc * p_cva

    # Calculate the likelihood of the initial state
    p_a = a_prob[initial_state[0]]
    p_b = b_prob[initial_state[1]]
    p_c = c_prob[initial_state[2]]
    p_avb = avb_prob[0][initial_state[0]][initial_state[1]]
    p_bvc = bvc_prob[initial_state[4]][initial_state[1]][initial_state[2]]
    p_cva = cva_prob[2][initial_state[2]][initial_state[0]]

    p_initial = p_a * p_b * p_c * p_avb * p_bvc * p_cva

    # Accept or reject candidate
    alpha = min(1, p_cand / p_initial)
    acceptance_criterion = random.uniform(0, 1)
    if acceptance_criterion < alpha:
        sample = candidate
    else:
        sample = initial_state
    return sample


def compare_sampling(bayes_net, initial_state):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""    
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    # TODO remove
    Gibbs_convergence = [0,0,0]
    N = 100
    delta = 0.0000001

    # Calculate Gibbs
    """
    cur_dist = np.array([0, 0, 0])
    prev_dist = np.array([0, 0, 0])
    current_state = initial_state
    convergence_counter = 0
    for i in range(1000000):
        new_state = Gibbs_sampler(bayes_net, current_state)
        current_state = new_state
        cur_dist[new_state[4]] += 1

        # Normalize cur and prev dists to get probabilities
        cur_normal = cur_dist / np.sum(cur_dist)
        prev_normal = prev_dist
        if np.sum(prev_dist) != 0:
            prev_normal = prev_dist / np.sum(prev_dist)
        diff = np.average(np.absolute(cur_normal - prev_normal))
        if diff <= delta:
            convergence_counter += 1
            if convergence_counter == N:
                Gibbs_count += 1
                break
        else:
            convergence_counter = 0
        prev_dist = np.copy(cur_dist)
        Gibbs_count += 1
    Gibbs_convergence = cur_dist / np.sum(cur_dist)
    """

    # Calculate MH
    N = 100
    delta = 0.00001
    cur_dist = np.array([0, 0, 0])
    prev_dist = np.array([0, 0, 0])
    current_state = initial_state
    convergence_counter = 0
    for i in range(1000000):
        candidate = MH_sampler(bayes_net, current_state)
        if candidate == current_state:
            MH_rejection_count += 1
        cur_dist[candidate[4]] += 1
        current_state = candidate
        # Normalize cur and prev dists to get probabilities
        cur_normal = cur_dist / np.sum(cur_dist)
        prev_normal = prev_dist
        if np.sum(prev_dist) != 0:
            prev_normal = prev_dist / np.sum(prev_dist)
        diff = np.average(np.absolute(cur_normal - prev_normal))
        if diff <= delta:
            convergence_counter += 1
            if convergence_counter == N:
                MH_count += 1
                break
        else:
            convergence_counter = 0
        prev_dist = np.copy(cur_dist)
        MH_count += 1
    MH_convergence = cur_dist / np.sum(cur_dist)

    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count


def sampling_question():
    """Question about sampling performance."""
    # TODO: assign value to choice and factor
    raise NotImplementedError
    choice = 2
    options = ['Gibbs','Metropolis-Hastings']
    factor = 0
    return options[choice], factor


def return_your_name():
    """Return your name from this function"""
    return "Nikhil Badami"
