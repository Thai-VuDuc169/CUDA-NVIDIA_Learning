#include <iostream>
#include <cassert>
#include<algorithm>
using namespace std;

int main()
{
   u_char root[10] = {1,2,5,6,7,10,8,4,3,9};
   u_char *a = root;
   if (*a < *(a+5))
   {
      cout << (int)*a << endl;
      cout << (int)(*(a+5)) << endl;
      
   }
   // sort(*a, *(a+10));
   // cout << root << endl;
   return 0;
}