"""
Author: Sonu Prasad
Email: sonu.prasad@mycit.ie
file: tsp.py
"""
import itertools
import random
import numpy as np
import os
import time
from numba import jit
from cities_graph import plot_graph
from utils import initialize_loggers
import multiprocessing


class TSP(object):
    def __init__(self, input_file_path, logger):
        self.input_file_path = input_file_path
        self.data = []
        self.random_solution = []
        self.total_cost = 0
        self.distance_mat = None
        self.logger = logger

    def read_instance(self):
        """
        Reading an instance from file path
        """
        input_file = open(self.input_file_path, 'r').read().splitlines()
        input_file.pop(0)
        cities = np.array([tuple(map(int, coord.split()[1:])) for coord in input_file])
        self.data = cities

    def generate_nearest_neighbour_solution(self):
        """
        Generate Nearest Neighbor Solution
        :return: void
        """
        cities = self.data.copy().tolist()
        random_selected_city = random.randint(0, len(cities) - 1)
        new_route = [cities[random_selected_city]]
        cities.pop(random_selected_city)

        while len(cities) >= 1:
            last_city = np.array(new_route[-1])
            pending_cities = np.array(cities)
            distances_arr = [self.euclidean_distance(last_city, c2) for c2 in pending_cities]
            min_dist_idx = np.argmin(distances_arr)
            nearest_city = pending_cities[min_dist_idx].tolist()
            new_route.append(nearest_city)
            cities.remove(nearest_city)

        self.random_solution = new_route
        self.total_cost = self.calc_tour_cost(new_route)

    @jit()
    def init_solution(self):
        """
        Generate random initial solution
        :return: void
        """
        copied_data = self.data.copy()
        data_len = copied_data.shape[0]
        for _ in range(data_len):
            n1 = random.randint(0, data_len - 1)
            n2 = random.randint(0, data_len - 1)

            copied_data[[n1,n2]] = copied_data[[n2,n1]]

        self.random_solution = copied_data.tolist()
        self.total_cost = self.calc_tour_cost(copied_data)

    @jit
    def calc_tour_cost(self, cities):
        """
        Calculate sum of euclidean distances between consecutive points including the first to last
        :return: total tour cost
        """
        cities_np_arr = np.array(cities)
        total_distance = 0.0
        for i in range(len(cities_np_arr)):
            total_distance += self.euclidean_distance(cities_np_arr[i - 1], cities_np_arr[i])
        return total_distance

    @staticmethod
    def euclidean_distance(c1, c2):
        """
        Distance between two cities
        :param c1: City 1
        :param c2: City 2
        :return: Euclidean distance between 2 cities
        """
        return np.linalg.norm(c2-c1)

    def generate_combinations(self, cities, node1, node2, node3):
        """
        This method generate 8 possible combinations from a list of cities
        :param cities: List of cities
        :param node1: [a, b]
        :param node2: [c, d]
        :param node3: [e, f]
        :return: Best combination i.e. list of cities, tour cost
        """
        """Combo 1: Same as the original node : Everything till Node 1 -> Node 1 to Node2 -> Node2 to Node 3 -> Node3 
        to Everything 
        """
        combo_1 = cities[:node1[0] + 1] + cities[node1[1]:node2[0] + 1] + cities[node2[1]: node3[0] + 1] + cities[node3[1]: ]
        combo_2 = cities[:node1[0] + 1] + cities[node1[1]:node2[0] + 1] + cities[node3[0]: node2[1] - 1: -1] + cities[node3[1]: ]
        combo_3 = cities[:node1[0] + 1] + cities[node2[0]:node1[1] - 1: -1] + cities[node2[1]: node3[0] + 1] + cities[node3[1]: ]
        combo_4 = cities[:node1[0] + 1] + cities[node2[0]:node1[1] - 1: -1] + cities[node3[0]: node2[1] - 1: -1] + cities[node3[1]: ]
        combo_5 = cities[:node1[0] + 1] + cities[node2[1]: node3[0] + 1] + cities[node1[1]:node2[0] + 1] + cities[node3[1]: ]
        combo_6 = cities[:node1[0] + 1] + cities[node2[1]: node3[0] + 1] + cities[node2[0]:node1[1] - 1: -1] + cities[node3[1]: ]
        combo_7 = cities[:node1[0] + 1] + cities[node3[0]: node2[1] - 1: -1] + cities[node1[1]:node2[0] + 1] + cities[node3[1]: ]
        combo_8 = cities[:node1[0] + 1] + cities[node3[0]: node2[1] - 1: -1] + cities[node2[0]:node1[1] - 1: -1] + cities[node3[1]: ]

        combinations_array = [combo_1, combo_2, combo_3, combo_4, combo_5, combo_6, combo_7, combo_8]
        distances_array = list(map(lambda x: self.calc_tour_cost(x), combinations_array))
        min_distance = int(np.argmin(distances_array))
        return combinations_array[min_distance], distances_array[min_distance]
        # self.random_solution = np.array(combinations_array[min_distance])
        # self.total_cost = distances_array[min_distance]

    def opt_3_local_search(self, route):
        """
        3 OPT Local search
        Generates all possible valid combinations.
        Runs a for loop for each combination obtained above and generates 7 different combinations
        possible after 3 OPT move. Selects the one with minimum tour cost
        :param route: list of cities
        :return: updated list of cities , tour_cost
        """
        all_combinations = list(itertools.combinations(range(len(route)), 3))
        """This generates all possible sorted routes and hence eliminating the need of for loop and then sorting it 
        and hence avoiding duplicates 
        """
        # Select any random city including first and last city
        random_city = np.random.randint(low=0, high=len(route))
        # Keep only valid combinations, i.e combinations containing the random selected city
        all_combinations = list(filter(lambda x: random_city in x, all_combinations))
        # Remove consecutive numbers to avoid overlaps and invalid cities
        # all_combinations = list(filter(lambda x: x[1] != x[0] + 1 and x[2] != x[1] + 1, all_combinations))

        for idx, item in enumerate(all_combinations):
            """
            Run for every combination generated above.
            a,c,e = x,y,z  # Generated in the combination
            d,e,f = x+1, y+1, z+1  # To form the edge
            """
            # print('Iteration count is {} and item a, c, e is {}' .format(idx, item))
            a1, c1, e1 = item
            b1, d1, f1 = a1+1, c1+1, e1+1

            """The above generates the edge. The edge is sent to generate 7 possible combinations and the best one is 
            selected and applied to the global solution
            """
            route, distance = self.generate_combinations(route, [a1, b1], [c1, d1], [e1, f1])

        distance = self.calc_tour_cost(route)
        return route, distance

    @jit()
    def perform_2_opt_swap(self):
        """
        Performs 2 opt swap 5 times
        Generates 2 random numbers a and b such that 0 <= a < len(cities) - 1 & a < b < len(cities)
        a: start index of the portion of the route to be reversed
        b: index of last node in portion of route to be reversed
        :return: Returns the new route created by 2opt swap i.e. list of cities
        """
        cities = self.random_solution.copy()
        size_of_cities = len(cities)
        self.logger.info('-------Running 2 Opt Perturbation 5 times-------')
        for i in range(5):
            c1, c2 = random.randrange(0, size_of_cities), random.randrange(0, size_of_cities)
            exclude = {c1}
            exclude.add(size_of_cities - 1) if c1 == 0 else exclude.add(size_of_cities - 1)
            exclude.add(0) if c1 == size_of_cities - 1 else exclude.add(c1 + 1)
            while c2 in exclude:
                c2 = random.randrange(0, size_of_cities)
            # to ensure we always have p1<p2
            if c2 < c1:
                c1, c2 = c2, c1

            assert 0 <= c1 < (size_of_cities - 1)
            assert c1 < c2 < size_of_cities

            cities[c1:c2] = reversed(cities[c1:c2])
        return cities

    def acceptance_criterion(self, best_found):
        """
        Compares the existing best solution with the solution obtained from perturbation and local search.
        Takes a Probability of 0.05 % for best_found otherwise compare it with the existing solution
        :param best_found: Best calculated after perturbation and local search
        :return: void
        """
        best_dist = self.calc_tour_cost(best_found)
        self.logger.info('Best Perturbed distance is {}'.format(best_dist))
        self.logger.info('Best solution available is {}'.format(self.total_cost))

        if random.random() < 0.05:
            self.random_solution = best_found
            self.total_cost = best_dist
        else:
            if best_dist < self.total_cost:
                self.random_solution = best_found
                self.total_cost = best_dist

    def main(self, initial_solution, file_name, iter_count):
        """
        This is the main function of TSP. Initial solution is generated based on value of initial_solution param.
        Runs 3 opt -- local search
        for 5 minutes:
            2 OPT
            local search
            acceptance criteria

        :param initial_solution: random or nn
        :param file_name: Essential for creating graph
        :param iter_count: Essential for creating graph
        :return: void
        """
        self.read_instance()
        if initial_solution == "nn":
            self.logger.info('Generating initial solution using nearest neighbour')
            self.generate_nearest_neighbour_solution()
        else:
            self.logger.info('Generating initial random solution')
            self.init_solution()

        """
        Plot Initial Graph
        """
        self.logger.info("---------- Initial cost is {} ----------".format(self.total_cost))
        plot_graph(self.random_solution, self.total_cost, 'Init_graph_{}_{}_{}'.format(file_name, initial_solution, iter_count))

        # Start 3 OPT Local Search
        opt_3_local_search_start_time = time.time()
        route, distance = self.opt_3_local_search(self.random_solution)
        self.random_solution = route
        self.total_cost = distance
        self.logger.info('--------3 OPT Local search completed in {}-------'.format(time.time() - opt_3_local_search_start_time))

        self.logger.info('Starting Perturbation -> Local Search -> Acceptance Criteria Phase for 5 minutes')
        # Perturbation phase
        seconds_to_run = 300
        counter = itertools.count()
        elapsed_time = time.time()
        while time.time() - elapsed_time < seconds_to_run:
            cities = self.perform_2_opt_swap()
            if time.time() - elapsed_time < seconds_to_run:
                best_perturbed_solution, distance = self.opt_3_local_search(cities)
                self.acceptance_criterion(best_perturbed_solution)
                next(counter)

        """
        Final Graph
        """
        plot_graph(self.random_solution, self.total_cost, 'final_graph_{}_{}_{}'.format(file_name, initial_solution, iter_count))
        self.logger.info('---------- Best Solution obtained is {} ----------'.format(self.total_cost))


def run(iter_count, ip_file_name):
    """
    For running TSP. This function is responsible for creating an object of TSP class and invoking the main function
    of the object
    :param iter_count: Iteration count for creating a log file with exact name
    :param ip_file_name: File Name
    :return: void
    """
    ip_file = os.path.join('dataset', 'Inst', ip_file_name)
    logger = initialize_loggers('tsp-{}'.format(iter_count))
    logger.info('------------Execution Count {}------------'.format(iter_count))
    tsp = TSP(ip_file, logger)
    tsp.main('random', 'inst-13.tsp', iter_count)


if __name__ == '__main__':
    """
    We will be using Python multiProcessing to spawn multiple processes to run TSP
    max iteration count is 5
    """
    file_name = 'inst-13.tsp'
    total_iteration_count = 0
    processes = []
    while total_iteration_count != 5:
        # log = initialize_loggers('tsp-{}'.format(total_iteration_count))
        # log.info('------------Execution Count {}------------'.format(total_iteration_count))
        # t = TSP(ip_file, log)
        # t.main('nn', 'inst-13.tsp', total_iteration_count)
        t = multiprocessing.Process(target=run, args=(total_iteration_count, file_name))
        total_iteration_count += 1
        processes.append(t)
        t.start()

    for process in processes:
        process.join()

    print("Done")
