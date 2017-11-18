#include <fstream.h>
#include <iostream>
#include <string.h>
#include <stdlib.h>

int factorial(int n) {
	int fact = 1;
	for (int i = 2; i <= n; ++i)
		fact *= i;
	return fact;
}

int combinaison(int k, int n) {
	int c = 1;
	int m = n - k;
	for (int i = n; i > m; i--)
		c *= i;
	return c / factorial(k);
}

int* get_combinations(int* arr_in, int k, int n) {
	int m = combinaison(k, n);
	int* comb = new int[m * k];
	bool fini = false;
	int id[k];
	int l = 0;
	for (int cptid = 0; cptid < k; cptid++)
		id[cptid] = cptid;
	id[k - 1] -=  1;
	while (!fini) {
		bool exit = false;
		for (int i = k - 1; i >= 0 && !exit; i--) {
			if (id[i] < n - k + i) {
				id[i]++;
				exit=true;
			} else {
				fini = (i == 0);
				bool first = true;
				for (int cpt = k - i; cpt > 0; cpt--) {
					id[k-cpt] = id[k - 1 - cpt] + (first ? 2 : 1);
					first = false;
				}
			}
		}
		if (!fini) {
			for (int cptid = 0; cptid < k; cptid++)
				comb[l * k + cptid] = arr_in[id[cptid]];
			l++;
		}
	}
	return comb;
}

void print_array(int* arr, int n, int m = 1) {
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j)
			std::cout << arr[i * n + j] << " ";
		std::cout << std::endl;
	}
}

int main(int argc, char** argv) {
	if (argc < 3) {
		std::cerr << "usage: " << argv[0] <<" k n [-v]" << std::endl;
		return 1;
	}
	int k = atoi(argv[1]);
	int n = atoi(argv[2]);
	bool verbose = false;
	if (argc > 3)
		verbose = (strcmp(argv[3], "-v") == 0);
	int arr[n];
	for (int i = 0; i < n; ++i) {
		arr[i] = i;
	}
	int* comb = get_combinations(arr, k, n);
	int m = combinaison(k, n);
	if (verbose) {
		std::cout << "full array\n";
		print_array(arr, n);
		std::cout << "all combinations\n";
		print_array(comb, k, m);
	}
	delete [] comb;
	return 0;
}
