#include <iostream>
#include <cassert>
using namespace std;

int* x()
{
   int *x = (int*)malloc(sizeof(int));
   *x = 9;
   return x;
};

int main()
{
   int *y = x();
   cout << &y << " : " << y << " : " << *y << endl;

   free(y);
   assert( *y != NULL);
   cout << &y << " : " << y << " : " << *y << endl;

   return 0;
}