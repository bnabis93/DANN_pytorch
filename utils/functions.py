from torch.autograd import Function

'''
autograd tutorial : https://pytorch.org/tutorials/beginner/former_torchies/autograd_tutorial.html

Autograd is now a core torch package for automatic differentiation.
Automatic differentiation!

example fucntions.
    gradient, backward, forward, ...


ctx
    similiar to class instance 'self'
    but we can not use 'self' bcz of static method!
    => use 'ctx'
    
    reference : https://stackoverflow.com/questions/49516188/difference-between-ctx-and-self-in-python
'''

class ReverseLayerF(Function):
    #Inheritance autograd's Function
    
    @staticmethod
    def forward(ctx, x,alpha):
        ctx.alpha = alpha
        
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        # neg -> reverse!
        output = grad_output.neg() * ctx.alpha
        return output, None
    
'''
What is staticmethod? (In korean)
    class, instance, static method 세가지가 존재한다.
    각 method들은 class 내부에서 정의된다.
    Instance method는 우리가 class를 사용할때 관습적으로 사용하는 'self'
    
    두 메소드 (class, static)은 상속시 큰 차이가 나고 그 특징이 두드러진다.
    
    Class method 
        @classmethod 라는 데코레이터를 통하여 선언된다.
        첫번째 인자를 cls로 주어야 한다.
        @classmethod
        def add (cls, x, y) :
            return x + y

        출처: https://hamait.tistory.com/635 [HAMA 블로그]
        
    Static method 
        @staticmethod 라는 데코레이터를 통하여 선언된다.
        우리가 흔히 아는 정적 메소드.
        객체별로 달라지는 것이 아닌, 공유하는 메소드
        정적 메소드 이기 때문에 self를 사용 할 수 없다.

'''