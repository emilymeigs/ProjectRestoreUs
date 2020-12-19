"""Vehicles Routing Problem (VRP)."""

# TODO: Maybe use this to assign houses to routes, but actual routing once assigned do something different?
# We also don't need them to return to the depot, maybe adding that in will help

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r*1000

def create_data_model():
    """Stores the data for the problem."""

    data = {}

    # use pandas to load in CSV
    # first row of CSV should be the location of the depot/pick-up

    # Load in data
    df = pd.read_csv("dec12_1.csv")

    # rename columns
    df.columns = ["Name", "Phone", "Address", "Apt", "City", "Position"]

    num_deliveries = df.shape[0]
    #num_vehicles = int(df.shape[0]/10+1)
    num_vehicles = 2
    distance_matrix = np.zeros((num_deliveries+1, num_deliveries+1)) # adding a fake depot for arbitrary start and end

    print(num_deliveries)

    x = []
    y = []

    # load in lat, lon and create distance matrix
    for index1, row1 in df.iterrows():
        address1 = str(row1["Position"]).split(", ")
        df.at[index1, "x"] = float(address1[0])
        df.at[index1, "y"] = float(address1[1])
        for index2, row2 in df.iterrows():
            address2 = str(row2["Position"]).split(", ")
            distance_matrix[index1][index2] = haversine(float(address1[0]), float(address1[1]), float(address2[0]), float(address2[1]))


    data['distance_matrix'] = distance_matrix
    data['num_vehicles'] = num_vehicles
    data['depot'] = num_deliveries
    demands = np.ones(num_deliveries+1)
    demands[num_deliveries] = 0 # no demand at fake "depot"
    data['demands'] = demands

    data['vehicle_capacities'] = [9, 8]

    return df, data


def print_solution(df, data, manager, routing, solution):
    """Prints solution on console."""
    df['Route Number'] = -1
    df['Stop Number'] = -1
    total_distance = 0
    total_load = 0
    for vehicle_id in range(data['num_vehicles']):
        stop_num = 1
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
            if index < 99:
                df.at[index, 'Route Number'] = vehicle_id
                df.at[index, 'Stop Number'] = stop_num
                stop_num += 1

        plan_output += ' {0} Load({1})\n'.format(manager.IndexToNode(index),
                                                 route_load)
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        plan_output += 'Load of the route: {}\n'.format(route_load)
        print(plan_output)
        total_distance += route_distance
        total_load += route_load
    print('Total distance of all routes: {}m'.format(total_distance))
    print('Total load of all routes: {}'.format(total_load))

    p1 = df.plot.scatter(x='x', y='y', c='Route Number', colormap='viridis')
    p1.plot()

    axes = plt.gca()
    axes.set_xlim([42.35, 42.45])
    axes.set_ylim([-71.14, -71.08])
    plt.show()


    df_sort = df.sort_values(by=['Route Number', 'Stop Number'])
    df_sort.to_csv('out_res.csv', index=False)


def main():
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    df, data = create_data_model()

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        9999999,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        print_solution(df, data, manager, routing, solution)


if __name__ == '__main__':
    main()