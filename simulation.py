import matplotlib.pyplot as plt
import random
import numpy as np
import operator
import statistics as stat
import math
import sys
import time
import pickle


timeslot_duration = 1e-3    # unit: sec


class Cell(object):
    def __init__(self, scheme, N):
        self.N = N
        self.ms_list = [MobileStation(i) for i in range(8)]
        num_class1_flow = int(((0.3 / 20e3) / ((0.3 / 20e3) + (0.4 / 200e3) + (0.3 / 400e3))) * N)
        num_class2_flow = int(((0.4 / 200e3) / ((0.3 / 20e3) + (0.4 / 200e3) + (0.3 / 400e3))) * N)
        num_class3_flow = int(((0.3 / 400e3) / ((0.3 / 20e3) + (0.4 / 200e3) + (0.3 / 400e3))) * N)
        if num_class1_flow == 0 and num_class2_flow == 0 and num_class3_flow == 0:
            num_class1_flow = N
        elif num_class1_flow != 0 and num_class2_flow == 0 and num_class3_flow == 0:
            num_class2_flow = N - num_class1_flow
        elif num_class1_flow != 0 and num_class2_flow != 0 and num_class3_flow == 0:
            num_class3_flow = N - num_class1_flow - num_class2_flow

        self.class1_flow_list = [Flow(i, 1) for i in range(num_class1_flow)]
        self.class2_flow_list = [Flow(i + num_class1_flow, 2) for i in range(num_class2_flow)]
        self.class3_flow_list = [Flow(i + num_class1_flow + num_class2_flow, 3) for i in range(num_class3_flow)]
        self.flow_list = self.class1_flow_list + self.class2_flow_list + self.class3_flow_list
        self.bs = BaseStation(scheme, self.class1_flow_list, self.class2_flow_list, self.class3_flow_list, self.ms_list)


class BaseStation(object):
    def __init__(self, scheme, class1_flow_list, class2_flow_list, class3_flow_list, ms_list):
        self.scheme = scheme
        self.class1_packets = []       # This list contains all the class1 users who have requested to transmit
        self.class2_packets = []       # This list contains all the class2 users who have requested to transmit
        self.class3_packets = []       # This list contains all the class3 users who have requested to transmit
        self.transmission_queue = []
        self.ms_list = ms_list
        self.downlink = 20e6
        self.transmitted_class1_packets = 0
        self.transmitted_class2_packets = 0
        self.transmitted_class3_packets = 0
        self.class1_flow_list = class1_flow_list
        self.class2_flow_list = class2_flow_list
        self.class3_flow_list = class3_flow_list
        self.num_class1_flow = len(self.class1_flow_list)
        self.num_class2_flow = len(self.class1_flow_list)
        self.num_class3_flow = len(self.class1_flow_list)
        self.class1_packet_delay = []
        self.class2_packet_delay = []
        self.class3_packet_delay = []
        self.class_packetdelay_matrix = [[] for i in range(3)]
        self.flow_list = self.class1_flow_list + self.class2_flow_list + self.class3_flow_list
        self.th_list = [0.0 for flow_i in range(len(self.flow_list))]
        self.WRR_timeslot_count = 0
        self.class_ms_throughput_matrix = [[0.0 for i in range(8)] for j in range(3)]
        self.class_ms_packetdelay_matrix = [[[] for i in range(8)] for j in range(3)]


    def categorize_packets(self):
        for flow in self.flow_list:
            if len(flow.packet_queue) > 0:
                if flow.class_type == 1:
                    self.class1_packets = self.class1_packets + flow.packet_queue
                    self.class1_packets = sorted(self.class1_packets, key=lambda x: x.flow_id)
                elif flow.class_type == 2:
                    self.class2_packets = self.class2_packets + flow.packet_queue
                    self.class2_packets = sorted(self.class2_packets, key=lambda x: x.flow_id)
                elif flow.class_type == 3:
                    self.class3_packets = self.class3_packets + flow.packet_queue
                    self.class3_packets = sorted(self.class3_packets, key=lambda x: x.flow_id)
                del flow.packet_queue[:]

    def transmit_packet(self):
        # if queue is not empty
        if len(self.transmission_queue) > 0:
            current_packet = self.transmission_queue[0]
            ms_id = current_packet.destination
            ms_downlink_rate = self.ms_list[ms_id].downlink_transmission_rate
            class1_spectral_resource = 0.3*(20e6)   # may need to fix this
            class2_spectral_resource = 0.4*(20e6)
            class3_spectral_resource = 0.3*(20e6)
            throughput = 0
            if current_packet.class_type == 1:
                throughput = ms_downlink_rate*class1_spectral_resource
            elif current_packet.class_type == 2:
                throughput = ms_downlink_rate*class2_spectral_resource
            elif current_packet.class_type == 3:
                throughput = ms_downlink_rate*class3_spectral_resource
            bits_per_timeslot = throughput*timeslot_duration
            current_packet.remaining_bits_to_transmit = current_packet.remaining_bits_to_transmit - bits_per_timeslot

            # if current packet finishes transmitting, remove it from the queue
            if current_packet.remaining_bits_to_transmit <= 0:
                if current_packet.class_type == 1:
                    self.transmitted_class1_packets = self.transmitted_class1_packets + 1
                    self.class_ms_throughput_matrix[0][current_packet.destination-1] = \
                    self.class_ms_throughput_matrix[0][current_packet.destination-1] + 480
                elif current_packet.class_type == 2:
                    self.transmitted_class2_packets = self.transmitted_class2_packets + 1
                    self.class_ms_throughput_matrix[1][current_packet.destination - 1] = \
                    self.class_ms_throughput_matrix[1][current_packet.destination - 1] + 1600
                elif current_packet.class_type == 3:
                    self.transmitted_class3_packets = self.transmitted_class3_packets + 1
                    self.class_ms_throughput_matrix[2][current_packet.destination - 1] = \
                    self.class_ms_throughput_matrix[2][current_packet.destination - 1] + 1200

                del self.transmission_queue[0]

        for p in self.class1_packets:
            p.time_delayed_in_queue = p.time_delayed_in_queue + timeslot_duration
        for p in self.class2_packets:
            p.time_delayed_in_queue = p.time_delayed_in_queue + timeslot_duration
        for p in self.class3_packets:
            p.time_delayed_in_queue = p.time_delayed_in_queue + timeslot_duration

    def assign_next_slot(self, current_timeslot):
        # if the current packet finish transmitting or queue is empty
        self.WRR_timeslot_count = self.WRR_timeslot_count + 1
        if len(self.transmission_queue) == 0:
            self.schedule(current_timeslot)
        elif self.transmission_queue[0].remaining_bits_to_transmit <= 0:
            self.schedule(current_timeslot)

    def schedule(self, current_timeslot):
        if self.scheme == 'PO':
            if len(self.class1_packets) > 0:
                # pick a class 1 packet
                packet_index = self.select_longest_delayed_packet(1)
                self.add_packet_to_queue(packet_index, 1)
            elif len(self.class2_packets) > 0:
                # pick a class 2 packet
                packet_index = self.select_longest_delayed_packet(2)
                self.add_packet_to_queue(packet_index, 2)
            elif len(self.class3_packets) > 0:
                # pick a class 3 packet
                packet_index = self.select_longest_delayed_packet(3)
                self.add_packet_to_queue(packet_index, 3)

        elif self.scheme == 'WRR':
            if self.select_class_WRR(current_timeslot) == 1 and len(self.class1_packets) > 0:
                self.add_packet_to_queue(0, 1)
            elif self.select_class_WRR(current_timeslot) == 2 and len(self.class2_packets) > 0:
                self.add_packet_to_queue(0, 2)
            elif self.select_class_WRR(current_timeslot) == 3 and len(self.class3_packets) > 0:
                self.add_packet_to_queue(0, 3)

        elif self.scheme == 'WRR+PFT':
            if self.select_class_WRR(current_timeslot) == 1 and len(self.class1_packets) > 0:
                packet_index = self.max_r2th_flow(1)
                self.add_packet_to_queue(packet_index, 1)
            elif self.select_class_WRR(current_timeslot) == 2 and len(self.class2_packets) > 0:
                packet_index = self.max_r2th_flow(2)
                self.add_packet_to_queue(packet_index, 2)
            elif self.select_class_WRR(current_timeslot) == 3 and len(self.class3_packets) > 0:
                packet_index = self.max_r2th_flow(3)
                self.add_packet_to_queue(packet_index, 3)

    # Select packet that has delayed the longest
    def select_longest_delayed_packet(self, class_type):
        packet_index = 0
        if class_type == 1:
            packet_index = self.class1_packets.index(max(self.class1_packets, key=lambda x: x.time_delayed_in_queue))
        elif class_type == 2:
            packet_index = self.class2_packets.index(max(self.class2_packets, key=lambda x: x.time_delayed_in_queue))
        elif class_type == 3:
            packet_index = self.class3_packets.index(max(self.class3_packets, key=lambda x: x.time_delayed_in_queue))
        return packet_index

    # select a class based on the WRR scheme
    def select_class_WRR(self, current_timeslot):
        selected_class = 1

        # calculating weight
        class1_throughput = 20e3  # bps
        class2_throughput = 200e3  # bps
        class3_throughput = 400e3  # bps
        class1_weight = int(class1_throughput / class1_throughput) * 25 * self.num_class1_flow
        class2_weight = int(class2_throughput / class1_throughput) * 25 * self.num_class2_flow
        class3_weight = int(class3_throughput / class1_throughput) * 25 * self.num_class3_flow
        wrr_cycle = class1_weight + class2_weight + class3_weight
        if len(self.class1_flow_list) > 0:
            if len(self.class2_flow_list) > 0:
                if len(self.class3_flow_list) == 0:
                    wrr_cycle = class1_weight + class2_weight
            else:
                if len(self.class3_flow_list) > 0:
                    wrr_cycle = class1_weight + class3_weight
                else:
                    wrr_cycle = class1_weight

        if self.WRR_timeslot_count <= class1_weight:
            if len(self.class1_packets) > 0:
                selected_class = 1
        else:
            if wrr_cycle == (class1_weight + class2_weight):
                if (self.WRR_timeslot_count > class1_weight) and (self.WRR_timeslot_count <= wrr_cycle):
                    selected_class = 2
            elif wrr_cycle == (class1_weight + class3_weight):
                if (self.WRR_timeslot_count > class1_weight) and (self.WRR_timeslot_count <= wrr_cycle):
                    selected_class = 3
            elif wrr_cycle == (class1_weight + class2_weight + class3_weight):
                if (self.WRR_timeslot_count > class1_weight) and (self.WRR_timeslot_count <= (class1_weight + class2_weight)):
                    selected_class = 2
                elif (self.WRR_timeslot_count > (class1_weight + class2_weight)) and (self.WRR_timeslot_count <= wrr_cycle):
                    selected_class = 3

        if self.WRR_timeslot_count >= wrr_cycle:
            self.WRR_timeslot_count = 0

        return selected_class

    # add a packet to the queue for transmission
    def add_packet_to_queue(self, packet_index, class_type):
        if class_type == 1:
            self.transmission_queue.append(self.class1_packets[packet_index])
            self.class_packetdelay_matrix[0].append(self.class1_packets[packet_index].time_delayed_in_queue)
            self.class_ms_packetdelay_matrix[0][self.class1_packets[packet_index].destination-1].append(
                self.class1_packets[packet_index].time_delayed_in_queue)
            del self.class1_packets[packet_index]
        elif class_type == 2:
            self.transmission_queue.append(self.class2_packets[packet_index])
            self.class_packetdelay_matrix[1].append(self.class2_packets[packet_index].time_delayed_in_queue)
            self.class_ms_packetdelay_matrix[1][self.class2_packets[packet_index].destination - 1].append(
                self.class2_packets[packet_index].time_delayed_in_queue)
            del self.class2_packets[packet_index]
        elif class_type == 3:
            self.transmission_queue.append(self.class3_packets[packet_index])
            self.class_packetdelay_matrix[2].append(self.class3_packets[packet_index].time_delayed_in_queue)
            self.class_ms_packetdelay_matrix[2][self.class3_packets[packet_index].destination - 1].append(
                self.class3_packets[packet_index].time_delayed_in_queue)
            del self.class3_packets[packet_index]

    # Calculates the sliding window average throughput rate
    def calculate_th(self, i, r):
        b = 0.8
        th = b*self.th_list[i] + (1-b)*r
        self.th_list[i] = th
        return th

    # Get data rate at current slot
    def get_r(self, class_type, ms_id):
        se_fraction = 0.3
        if class_type == 2:
            se_fraction = 0.4
        r = se_fraction * 20e6 * self.ms_list[ms_id].downlink_transmission_rate
        return r

    # Pick a packet from a flow within a list that maximizes the ratio r(i,k)/TH(i,k)
    def max_r2th_flow(self, class_type):
        packet_list = []
        if class_type == 1:
            packet_list = self.class1_packets
        elif class_type == 2:
            packet_list = self.class2_packets
        elif class_type == 3:
            packet_list = self.class3_packets

        r2th_ratio_list = []
        for p in packet_list:
            r = self.get_r(class_type, p.destination)
            th = self.calculate_th(p.flow_id, r)
            r2th_ratio = r/th
            r2th_ratio_list.append(r2th_ratio)
        max_r2th_index = r2th_ratio_list.index(max(r2th_ratio_list))
        return max_r2th_index


class MobileStation(object):
    def __init__(self, id):
        self.id = id

        # Assign downlink transmission rate according to the following distribution:
        # 40% 0.2 bps/Hz    30% 1bps/Hz     30% 2bps/Hz
        self.downlink_transmission_rate = 0  # unit is bps/Hz
        dice = random.random()
        if dice < 0.4:
            self.downlink_transmission_rate = 0.2
        elif (0.4 <= dice) and (dice < 0.7):
            self.downlink_transmission_rate = 1
        elif (0.7 <= dice):
            self.downlink_transmission_rate = 2


class Flow(object):
    def __init__(self, flow_id, class_type):
        self.flow_id = flow_id
        self.ms_id = random.randint(0, 7)
        self.class_type = class_type
        self.packet_queue = []
        self.status = 'pause'
        self.lag = 0    # unit: timeslots
        self.packet_generation_timer = 0
        self.activity_burst_timer = 0
        self.pause_timer = 0

        if self.class_type == 1:
            self.lag = random.randint(0, 600-1)
        elif self.class_type == 2:
            self.lag = random.randint(0, int(4/timeslot_duration)-1)
        elif self.class_type == 3:
            self.lag = random.randint(0, 2)
            self.status = 'Poisson'

    def generate_flow(self, current_timeslot):
        # if there is lag in the beginning, decrement lag
        if self.lag > 0:
            self.lag = self.lag - 1
        else:
            if self.class_type == 1 or self.class_type == 2:
                if self.status == 'pause':
                    # if from pause to activity burst, reset activity burst timer
                    if self.pause_timer <= 0:
                        self.status = 'activity_burst'
                        if self.class_type == 1:
                            self.activity_burst_timer = 400
                            self.packet_generation_timer = 0
                        elif self.class_type == 2:
                            self.activity_burst_timer = 1000
                            self.packet_generation_timer = 0
                    else:
                        # continue in the pause status
                        self.pause_timer = self.pause_timer - 1
                # if already in activity burst
                elif self.status == 'activity_burst':
                    if self.activity_burst_timer > 0:
                        if self.packet_generation_timer > 0:
                            self.packet_generation_timer = self.packet_generation_timer - 1
                        else:
                            if self.class_type == 1:
                                self.packet_queue.append(Packet(480, 1, self.flow_id, self.ms_id))
                                # reset generation timer
                                self.packet_generation_timer = int((480/20e3)/timeslot_duration)
                            elif self.class_type == 2:
                                self.packet_queue.append(Packet(1600, 2, self.flow_id, self.ms_id))
                                # reset generation timer
                                self.packet_generation_timer = int((1600/200e3)/timeslot_duration)
                        self.activity_burst_timer = self.activity_burst_timer - 1
                    else:
                        # activity burst going into pause
                        self.status = 'pause'
                        if self.class_type == 1:
                            self.pause_timer = 600
                        elif self.class_type == 2:
                            self.pause_timer = 4000
            elif self.class_type == 3:
                # Generate packet according Poisson process
                interpacket_arrival = int(1200/400)
                if current_timeslot % interpacket_arrival == 0:
                    self.packet_queue.append(Packet(1200, 3, self.flow_id, self.ms_id))


class Packet(object):
    def __init__(self, size, class_type, flow_id, destination):
        self.size = size
        self.destination = destination
        self.time_delayed_in_queue = 0
        self.remaining_bits_to_transmit = self.size
        self.class_type = class_type
        self.flow_id = flow_id


class Simulator(object):
    def __init__(self):
        self.N_array = []
        self.total_throughput_array = []
        self.class1_throughput_array = []
        self.class2_throughput_array = []
        self.class3_throughput_array = []
        self.class1_avg_packetdelay_array = []
        self.class2_avg_packetdelay_array = []
        self.class3_avg_packetdelay_array = []
        self.class1_std_packetdelay_array = []
        self.class2_std_packetdelay_array = []
        self.class3_std_packetdelay_array = []
        self.class_ms_throughput_matrix_list = []
        self.class_ms_avg_packetdelay_matrix_list = []
        self.class_ms_std_throughput_list = []
        self.class_ms_std_packetdelay_list = []

    def simulate(self, scheme, N, minutes):
        print('Simulating N = ', str(N))
        cell = Cell(scheme, N)

        num_timeslots = int((minutes*60)/timeslot_duration)

        # Start time simulation
        for k in range(num_timeslots):
            # Generate flows
            for flow in cell.flow_list:
                flow.generate_flow(k)

            # Base Station should categorize these packets by class types
            cell.bs.categorize_packets()
            # Get transmit current packet if any
            cell.bs.transmit_packet()
            # Assign next time slot
            cell.bs.assign_next_slot(k)

        simulation_duration = num_timeslots*timeslot_duration

        # Calculate throughput
        class_ms_throughput_matrix = np.divide(np.matrix(cell.bs.class_ms_throughput_matrix), simulation_duration * 1e6)
        self.class_ms_throughput_matrix_list.append(class_ms_throughput_matrix)
        class1_throughput = np.sum(class_ms_throughput_matrix[0])
        class2_throughput = np.sum(class_ms_throughput_matrix[1])
        class3_throughput = np.sum(class_ms_throughput_matrix[2])
        total_throughput = class1_throughput + class2_throughput + class3_throughput
        self.class1_throughput_array.append(class1_throughput)
        self.class2_throughput_array.append(class2_throughput)
        self.class3_throughput_array.append(class3_throughput)
        self.total_throughput_array.append(total_throughput)
        class_ms_std_throughput = [0.0 for j in range(3)]
        for j in range(3):
            class_ms_std_throughput[j] = np.std(class_ms_throughput_matrix[j])
        self.class_ms_std_throughput_list.append(class_ms_std_throughput)

        # Calculate packet delay
        for j in range(3):
            for i in range(8):
                if len(cell.bs.class_ms_packetdelay_matrix[j][i]) == 0:
                    cell.bs.class_ms_packetdelay_matrix[j][i].append(0.0)
        class_ms_avg_packetdelay_matrix = np.matrix(
            [[stat.mean(cell.bs.class_ms_packetdelay_matrix[j][i]) for i in range(8)] for j in range(3)])
        self.class_ms_avg_packetdelay_matrix_list.append(class_ms_avg_packetdelay_matrix)

        class_ms_std_packetdelay = [0.0 for j in range(3)]
        for j in range(3):
            class_ms_std_packetdelay[j] = np.std(class_ms_avg_packetdelay_matrix[j])

        class_std_packetdelay = [0.0 for i in range(3)]
        for j in range(3):
            if len(cell.bs.class_packetdelay_matrix[j]) == 0:
                cell.bs.class_packetdelay_matrix[j].append(0.0)
            if len(cell.bs.class_packetdelay_matrix[j]) > 1:
                class_std_packetdelay[j] = stat.stdev(cell.bs.class_packetdelay_matrix[j])
        class1_avg_packetdelay = stat.mean(cell.bs.class_packetdelay_matrix[0])
        class2_avg_packetdelay = stat.mean(cell.bs.class_packetdelay_matrix[1])
        class3_avg_packetdelay = stat.mean(cell.bs.class_packetdelay_matrix[2])
        self.class1_avg_packetdelay_array.append(class1_avg_packetdelay)
        self.class2_avg_packetdelay_array.append(class2_avg_packetdelay)
        self.class3_avg_packetdelay_array.append(class3_avg_packetdelay)
        self.class1_std_packetdelay_array.append(class_std_packetdelay[0])
        self.class2_std_packetdelay_array.append(class_std_packetdelay[1])
        self.class3_std_packetdelay_array.append(class_std_packetdelay[2])


    def run(self, scheme, N_total):
        start = time.clock()
        simulation_time = 5     # in minutes
        for i in range(N_total):
            N = i + 1
            self.N_array.append(N)
            self.simulate(scheme, N, simulation_time)

        end = time.clock()
        print('\nSimulation ended. Program run-time in your CPU:', str(int(end - start)), 'seconds.')


    def save_simulation_data(self, file_name):
        simulatior_pickle = open(file_name, 'wb')
        pickle.dump(self, simulatior_pickle)
        simulatior_pickle.close()


class Unpacker(object):
    def extract(self, pickle_filename):
        sim_pickle = open(pickle_filename, 'rb')
        simulator = pickle.load(sim_pickle)

        # Plot throughput rates
        plt.plot(simulator.N_array, simulator.total_throughput_array, marker='o', linestyle='-')
        plt.plot(simulator.N_array, simulator.class1_throughput_array, marker='o', linestyle='-')
        plt.plot(simulator.N_array, simulator.class2_throughput_array, marker='o', linestyle='-')
        plt.plot(simulator.N_array, simulator.class3_throughput_array, marker='o', linestyle='-')
        plt.title('Throughput vs. N')
        plt.ylabel('throughput (Mbps)')
        plt.xlabel('N')
        plt.legend(['total', 'class 1', 'class 2', 'class 3'])
        plt.show()

        # Plot average packet delays for different classes
        plt.plot(simulator.N_array, np.multiply(simulator.class1_avg_packetdelay_array, 1000), marker='o', linestyle='-', color='orange')
        plt.title('Class-1 Flow Average Packet Delay vs. N')
        plt.ylabel('average packet delay (ms)')
        plt.xlabel('N')
        plt.show()

        plt.plot(simulator.N_array, np.multiply(simulator.class2_avg_packetdelay_array, 1000), marker='o', linestyle='-', color='g')
        plt.title('Class-2 Flow Packet Delay vs. N')
        plt.ylabel('average packet delay (ms)')
        plt.xlabel('N')
        plt.show()

        plt.plot(simulator.N_array, simulator.class3_avg_packetdelay_array, marker='o', linestyle='-', color='r')
        plt.title('Class-3 Flow Packet Delay vs. N')
        plt.ylabel('average packet delay (s)')
        plt.xlabel('N')
        plt.show()

        # Plot standard deviation packet delays for different classes
        plt.plot(simulator.N_array, np.multiply(simulator.class1_std_packetdelay_array, 1000), marker='o',
                 linestyle='-', color='orange')
        plt.title('Class-1 Flow Standard Deviation of Packet Delay vs. N')
        plt.ylabel('standard deviation of packet delay (ms)')
        plt.xlabel('N')
        plt.show()

        plt.plot(simulator.N_array, np.multiply(simulator.class2_std_packetdelay_array, 1000), marker='o',
                 linestyle='-', color='g')
        plt.title('Class-2 Flow Standard Deviation of Packet Delay vs. N')
        plt.ylabel('standard deviation of packet delay (ms)')
        plt.xlabel('N')
        plt.show()

        plt.plot(simulator.N_array, simulator.class3_std_packetdelay_array, marker='o', linestyle='-', color='r')
        plt.title('Class-3 Flow Standard Deviation of Packet Delay vs. N')
        plt.ylabel('standard deviation of packet delay (s)')
        plt.xlabel('N')
        plt.show()

        a = np.matrix(simulator.class_ms_std_throughput_list)
        plt.plot(simulator.N_array, a[:, 0], marker='o', linestyle='-', color='orange')
        plt.title('Class-1 Standard Deviation of Throughput over mobile stations vs. N')
        plt.ylabel('standard deviation of throughput over mobile stations (Mbps)')
        plt.xlabel('N')
        plt.show()

        plt.plot(simulator.N_array, a[:, 1], marker='o', linestyle='-', color='g')
        plt.title('Class-2 Standard Deviation of Throughput over mobile stations vs. N')
        plt.ylabel('standard deviation of throughput over mobile stations (Mbps)')
        plt.xlabel('N')
        plt.show()

        plt.plot(simulator.N_array, a[:, 2], marker='o', linestyle='-', color='r')
        plt.title('Class-3 Standard Deviation of Throughput over mobile stations vs. N')
        plt.ylabel('standard deviation of throughput over mobile stations (Mbps)')
        plt.xlabel('N')
        plt.show()

        for i in range(len(simulator.N_array)):
            class1_std = np.std(np.matrix(simulator.class_ms_avg_packetdelay_matrix_list[i])[0, :])
            class2_std = np.std(np.matrix(simulator.class_ms_avg_packetdelay_matrix_list[i])[1, :])
            class3_std = np.std(np.matrix(simulator.class_ms_avg_packetdelay_matrix_list[i])[2, :])
            simulator.class_ms_std_packetdelay_list.append([class1_std, class2_std, class3_std])

        b = np.matrix(simulator.class_ms_std_packetdelay_list)
        plt.plot(simulator.N_array, b[:, 0], marker='o', linestyle='-', color='orange')
        plt.title('Class-1 Standard Deviation of Packet Delay over Mobile Stations vs. N')
        plt.ylabel('standard deviation of packet delay over mobile stations (s)')
        plt.xlabel('N')
        plt.show()

        plt.plot(simulator.N_array, b[:, 1], marker='o', linestyle='-', color='g')
        plt.title('Class-2 Standard Deviation of Packet Delay over Mobile Stations vs. N')
        plt.ylabel('standard deviation of packet delay over mobile stations (s)')
        plt.xlabel('N')
        plt.show()

        plt.plot(simulator.N_array, b[:, 2], marker='o', linestyle='-', color='r')
        plt.title('Class-3 Standard Deviation of Packet Delay over Mobile Stations vs. N')
        plt.ylabel('standard deviation of packet delay over mobile stations (s)')
        plt.xlabel('N')
        plt.show()


if __name__ == "__main__":
    tot_start = time.clock()

    # Simulate with the 3 schemes
    # Priority Oriented
    print('Simulating priority oriented scheme.')
    sim1 = Simulator()
    sim1.run(scheme='PO', N_total=40)
    sim1.save_simulation_data('PO_simulation_data.pcy')
    del sim1

    # Weighted Round Robin
    print('Simulating WRR scheme.')
    sim2 = Simulator()
    sim2.run(scheme='WRR', N_total=15)
    sim2.save_simulation_data('WRR_simulation_data.pcy')
    del sim2

    # WRR + PFT
    print('Simulating WRR+PFT scheme.')
    sim3 = Simulator()
    sim3.run(scheme='WRR+PFT', N_total=15)
    sim3.save_simulation_data('WRR+PFT_simulation_data.pcy')
    del sim3

    # Unpack and extract the simulation data saved in pickle files
    u = Unpacker()
    u.extract('PO_simulation_data.pcy')
    u.extract('WRR_simulation_data.pcy')
    u.extract('WRR+PFT_simulation_data.pcy')

    tot_end = time.clock()
    print('\nDone. Program run-time in your CPU:', str(int(tot_end - tot_start)), 'seconds.')
