#include<string.h>
#include<stdio.h>
#include<stdlib.h>

#define LEN sizeof(words)
#define LENS sizeof(sents)

struct words{
	char word[80];
	struct words * next;
};
struct sents{
	struct words * ans;
	struct sents * next;
};
long int n=0;
int devideHz(char sen[], char han[1000][4])//把一句话切分成一个一个汉字,在此假定一句话不超过1000个字
{
	int num=0;
	int i=0;
	while(i< strlen(sen))
	{
		if ((sen[i] & 0x80) == 0)
		{
			han[num][0]=sen[i];
			han[num][1]='\0';
			i++;
			num++;
		}
		else if((sen[i] & 0xE0) == 0xC0)
		{
			han[num][0]=sen[i];
			han[num][1]=sen[i+1];
			han[num][2]='\0';
			i+=2;
			num++;
		}
		else if((sen[i] & 0xF0) == 0xE0)
		{
			han[num][0]=sen[i];
			han[num][1]=sen[i+1];
			han[num][2]=sen[i+2];
			han[num][3]='\0';
			i+=3;
			num++;
		}
		else
		{
			return -1;//not decoding in utf-8
		}
	}
	if (i !=strlen(sen))
	{
		return -1;//not decoding in utf-8
	}
	return num;
}
int compareInOneSent(struct words * res,struct words * ans){
	int count=0;	//用于存放句子中正确识别的个数
	struct words * p,*q;
	p=res;
	q=ans;
	int pointP=0,pointQ=0;//用于标记这个字是句子中第几个字，以判断是哪个指针需要next操作
	while(p!=NULL&&q!=NULL){
		char temp[1000][4];
		int pp=devideHz(p->word,temp);
		int qq=devideHz(q->word,temp);
		if(strcmp(p->word,q->word)==0 &&pointP==pointQ){
			count++;
			p=p->next;
			q=q->next;
			pointP+=pp;
			pointQ+=qq;
		}
		else{
			if((pointP+pp)>(pointQ+qq)){
				q=q->next;
				pointQ+=qq;
			}
			else	
				if((pointP+pp)<(pointQ+qq)){
					p=p->next;
					pointP+=pp;
				}
				else{
					p=p->next;
					q=q->next;
					pointP+=pp;
					pointQ+=qq;
				}
		}
	}
	return count;
}
int judgeFirst(struct words * head,char word[]){
	if(head==NULL)
		return 0;
	struct words * p=head;
	for(int i=0;i<n;i++){
		if(strcmp(p->word,word)==0)
			return -1;
		p=p->next;
	}
	return 0;
}
int judgeSame(struct words * head,char word[]){
	if(head==NULL)
		return 0;
	struct words * p=head;
	while(p!=NULL){
		if(strcmp(p->word,word)==0)
			return -1;
		p=p->next;
	}
	return 0;
}
struct words * genDic(char file[]){
	FILE * fp;
	if((fp=fopen(file,"r"))==NULL){
		return NULL;
	}
	struct words * head,*p,*q;
	head=NULL;
	while(!feof(fp)){
		char temp[80];
		p=(struct words *)malloc(LEN);
		int i=fscanf(fp,"%s%s%s%s%s%s%s%s%s%s",temp,p->word,temp,temp,temp,temp,temp,temp,temp,temp);
		if(i==-1)
			continue;
		if(judgeFirst(head,p->word)==0){
			if(head==NULL)
				head=p;
			else
				q->next=p;
			q=p;
			n++;
		}
	}
	q->next=NULL;
	return head;
}
struct words * MM(struct words *head,char sen[],int win){
	if(win<1)
		return NULL;
	char hz[1000][4];
	int num=devideHz(sen,hz);
	int point=0;
	struct words * res,*p,*q;
	res=NULL;
	while(point<num){
		int size=win;
		while(num<size+point){
			size--;
		}
		while(size>1){
			char temp[80]="\0";
			for(int i=0;i<size;i++){
				strcat(temp,hz[i+point]);
			}
			if(judgeSame(head,temp)==-1){
				p=(struct words *)malloc(LEN);
				strcpy(p->word,temp);
				if(res==NULL)
					res=p;
				else
					q->next=p;
				q=p;
				point=point+size;
				break;
			}
			else{
				size--;
			}
		}
		if(size==1){
			p=(struct words *)malloc(LEN);
			strcpy(p->word,hz[point]);
			if(res==NULL)
				res=p;
			else
				q->next=p;
			q=p;
			point=point+1;
		}	
	}
	q->next=NULL;
	return res;
}
struct words * RMM(struct words *head,char sen[],int win){
	if(win<1)
		return NULL;
	char hz[1000][4];
	int num=devideHz(sen,hz);
	int point=num;
	struct words * res,*p,*q;
	res=p=q=NULL;
	while(point>=1){
		int size=win;
		while(0>point-size){
			size--;
		}
		while(size>1){
			char temp[80]="\0";
			for(int i=point-size;i<point;i++){
				strcat(temp,hz[i]);
			}
			if(judgeSame(head,temp)==-1){
				p=(struct words *)malloc(LEN);
				strcpy(p->word,temp);
				if(res==NULL)
					res=p;
				else
					q->next=p;
				q=p;
				point=point-size;
				break;
			}
			else{
				size--;
			}
		}
		if(size==1){
			p=(struct words *)malloc(LEN);
			strcpy(p->word,hz[point-size]);
			if(res==NULL)
				res=p;
			else
				q->next=p;
			q=p;
			point=point-1;
		}	
	}
	if(q!=NULL)
	q->next=NULL;
	return res;
}
struct words * getDicFromTxt(char file[]){
	FILE * fp;
	if((fp=fopen(file,"r"))==NULL){
		return NULL;
	}
	int count=0;
	struct words * head,*p,*q;
	head=NULL;
	while(!feof(fp)){
		p=(struct words *)malloc(LEN);
		int i=fscanf(fp,"%s",p->word);
		if(i==-1)
			continue;
		if(head==NULL)
			head=p;
		else
			q->next=p;
		q=p;
		count++;
	}
	q->next=NULL;
	fclose(fp);
//	printf("词典数目为%d\n",count);
	return head;
}
int outputP(struct words * head,FILE * fp){
	struct words * p=head;
	while(p!=NULL){
		fputs(p->word,fp);
		fputs("\n",fp);
		p=p->next;
	}
	return 0;
}
int outputDic(struct words * head,char file[]){// 将字典输出
	FILE * fp;
	if((fp=fopen(file,"w+"))==NULL){
		return -1;
	}
	struct words * p=head;
	while(p!=NULL){
		fputs(p->word,fp);
		fputs("\n",fp);
		p=p->next;
	}
	return 0;
}
struct sents * outPutSentFile(char file[]){//将句子读出
	FILE * fp;
	if((fp=fopen(file,"r"))==NULL){
		return NULL ;
	}
	char fir[80]="\0";
	int isFirst=0;
	struct words * head,*p,*q;
	head=NULL;
	struct sents * headS,*pS,*qS;
	headS=NULL;
	while(!feof(fp)){
		char temp[80];
		char word[80];
		char num[80];	
		int i=fscanf(fp,"%s%s%s%s%s%s%s%s%s%s",num,word,temp,temp,temp,temp,temp,temp,temp,temp);
		if(i==-1)
			continue;
		if(strcmp(num,fir)==0&&isFirst==-1){//一句话结束
			q->next=NULL;
			pS=(struct sents *)malloc(LENS);
			pS->ans=head;
			if(headS==NULL)
				headS=pS;
			else
				qS->next=pS;
			qS=pS;
			head=NULL;
		}
		if(isFirst==0){//从文件中读入utf-8的1的字符串
			strcpy(fir,num);
			isFirst=-1;
		}
		p=(struct words *)malloc(LEN);
		strcpy(p->word,word);
		if(head==NULL)
			head=p;
		else
			q->next=p;
		q=p;	
	}
	q->next=NULL;
	pS=(struct sents *)malloc(LENS);
	pS->ans=head;
	if(headS==NULL)
		headS=pS;
	else
		qS->next=pS;
	qS=pS;
	head=NULL;
	qS->next=NULL;
	return headS;
}
int sizeOfPoint(struct words * head){
	struct words * p=head;
	int count=0;
	while(p!=NULL){
		count++;
		p=p->next;
	}
	return count;
}
struct words * reverse(struct words * head){
	struct words * p=head;
	if(p==NULL){
		return NULL;
	}
	struct words * res,*q,*temp;;
	res=NULL;
	while(p!=NULL){
		if(res==NULL){
			res=p;	
			temp=p->next;
			res->next=NULL;
		}
		else{
			q=p;
			temp=p->next;
			q->next=res;
			res=q;
		}
		p=temp;
	}
	return res;
}
int main(){
//	char file[]="data.conll";
//	struct words * head =genDic(file);
//	printf("%d\n",n);
//	char dic[]="dic.txt";
//	outputDic(head,dic);
	//上面注释掉的语句用来第一次从conll文件中生成词典

	char dic[]="word.dict";
	struct words * head =getDicFromTxt(dic);
	//这段代码用于从文本中读入词典
	FILE * fp;
	if((fp=fopen("res.txt","w+"))==NULL){
		return 0;
	}

	struct sents * headS=outPutSentFile("data.conll");
	struct sents * p=headS;	
	long int countA=0,countR=0,countP=0;
	int i=0;
	while(p!=NULL){
		struct words * q=NULL;
		q=p->ans;
		char sent[10000]="\0";

		while(q!=NULL){
			strcat(sent,q->word);
			q=q->next;
		}
		struct words * res1=MM(head,sent,3);//此段注释掉的内容是正向最大匹配的代码
		outputP(res1,fp);
		countR+=sizeOfPoint(res1);
		countA+=sizeOfPoint(p->ans);
		countP+=compareInOneSent(res1,p->ans);
/*		struct words * res1=RMM(head,sent,3);
		struct words * res2=reverse(res1);
		outputP(res2,fp);
		countR+=sizeOfPoint(res2);
		countA+=sizeOfPoint(p->ans);
		countP+=compareInOneSent(res2,p->ans);
*/
		p=p->next;
		i++;
//		printf("%d\n",i);
	}
	
//	printf("R %d\t\tP %d\t\tA %d\n",countR,countP,countA);
	float P = (float)countP/(float)countR;
	float R = (float)countP/(float)countA;
	float F = R*P*2/(P+R);
	printf("正确率：%f\n召回率：%f\nF值：%f\n",P,R,F);
    return 0;
}

