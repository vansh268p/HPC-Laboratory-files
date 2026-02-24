[202301163@gics0 Lab_3_202301163-419]$ g++ init.cpp main.cpp utils.cpp -o main.out -O2
[202301163@gics0 Lab_3_202301163-419]$ ./main.out input_a.bin
Total interpolation time (serial) = 0.250000 seconds
[202301163@gics0 Lab_3_202301163-419]$ ./main.out input_b.bin
Total interpolation time (serial) = 1.200000 seconds
[202301163@gics0 Lab_3_202301163-419]$ ./main.out input_c.bin
Total interpolation time (serial) = 0.880000 seconds
[202301163@gics0 Lab_3_202301163-419]$ ./main.out input_d.bin
Total interpolation time (serial) = 4.630000 seconds
[202301163@gics0 Lab_3_202301163-419]$ ./main.out input_e.bin
Total interpolation time (serial) = 3.430000 seconds
[202301163@gics0 Lab_3_202301163-419]$


| Config | NX   | NY  | Grid Points | Particles  | Your Time (s) 		        |
| ------ | ---- | --- | ----------- | ---------- | -----------------------------|
| (a)    | 250  | 100 | 25,351      | 900,000    | 0.250000 seconds             |
| (b)    | 250  | 100 | 25,351      | 5,000,000  | 1.200000 seconds             |
| (c)    | 500  | 200 | 100,701     | 3,600,000  | 0.880000 seconds             |
| (d)    | 500  | 200 | 100,701     | 20,000,000 | 4.630000 seconds             |
| (e)    | 1000 | 400 | 401,401     | 14,000,000 | 3.430000 seconds             |
