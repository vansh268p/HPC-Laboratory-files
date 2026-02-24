| Config | NX   | NY  | Grid Points | Particles  | Your Time (s) 		        |
| ------ | ---- | --- | ----------- | ---------- | -----------------------------|
| (a)    | 250  | 100 | 25,351      | 900,000    | 0.042724 seconds             |
| (b)    | 250  | 100 | 25,351      | 5,000,000  | 0.227670 seconds             |
| (c)    | 500  | 200 | 100,701     | 3,600,000  | 0.170778 seconds             |
| (d)    | 500  | 200 | 100,701     | 20,000,000 | 0.919918 seconds             |
| (e)    | 1000 | 400 | 401,401     | 14,000,000 | 1.040558 seconds             |

student@student-HP-Pro-Tower-280-G9-PCI-Desktop-PC:~/Documents/HPC_LAB/Assignment 03$ ./main.out input_a.bin
Total interpolation time (serial) = 0.042724 seconds
student@student-HP-Pro-Tower-280-G9-PCI-Desktop-PC:~/Documents/HPC_LAB/Assignment 03$ ./main.out input_b.bin
Total interpolation time (serial) = 0.227670 seconds
student@student-HP-Pro-Tower-280-G9-PCI-Desktop-PC:~/Documents/HPC_LAB/Assignment 03$ ./main.out input_c.bin
Total interpolation time (serial) = 0.170778 seconds
student@student-HP-Pro-Tower-280-G9-PCI-Desktop-PC:~/Documents/HPC_LAB/Assignment 03$ ./main.out input_d.bin
Total interpolation time (serial) = 0.919918 seconds
student@student-HP-Pro-Tower-280-G9-PCI-Desktop-PC:~/Documents/HPC_LAB/Assignment 03$ ./main.out input_e.bin
Total interpolation time (serial) = 1.040558 seconds
