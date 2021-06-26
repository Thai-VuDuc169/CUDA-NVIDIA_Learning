#include <iostream>
#include <cassert>
using namespace std;

int* x()
{
   // int *x = (int*)malloc(sizeof(int));
   int *x = new int;
   return x;
};

int main()
{
   int *y = x();
   *y = 1;
   cout << &y << " : " << y << " : " << *y << endl;

   delete (y);
   // assert( *y != NULL);
   cout << &y << " : " << y << " : " << *y << endl;

   return 0;
}