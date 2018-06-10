#include <stdio.h>

int main()
{
    FILE * fp = fopen("data/test-UTF8","r");
    char ch = getc(fp);
    while (ch != EOF)
    {
        int test = ch & 0xf0;
        if(test >= 0xf0)
        {
            printf("%c",ch);
            for(int i=0 ; i<3 ; i++)
            {
                ch = getc(fp);
                printf("%c",ch);
            }
            printf("  ");
            ch = getc(fp);
            
        }
        else if(test >= 0xe0 && test < 0xf0){
            printf("%c",ch);
            for(int i=0 ; i<2 ; i++)
            {
                ch = getc(fp);
                printf("%c",ch);
            }
            printf("  ");
            ch = getc(fp);
            continue;
        }
        else if(test >= 0xc0 && test <0xe0){
            printf("%c",ch);
            ch = fgetc(fp);
            printf("%c  ",ch);
            ch = fgetc(fp);
            continue;
        }
        else
        {
            printf("%c  ",ch);
            ch = getc(fp);
        }
    }
    printf("\n");
    return 0;
}

