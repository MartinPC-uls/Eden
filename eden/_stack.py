import eden

class Stack:
    '''
    A stack can contain either 'Matrix' or 'Stack' types.
    
    Similar to a Tensor, a stack has dimensions depending on how 'items'
    is being used. You can only create a Stack of stacks, or a Stack of matrices.
    '''
    def __init__(self, items=[]):
        self.items = []
        self.stack_type = None
        if len(items) > 0:
            self.stack_type = type(items[0])
            
        for element in items:
            if type(element) != self.stack_type:
                raise RuntimeError("You can only create a Stack for one type, either Matrix or Stack.")
            
            self.items.append(element)
            
    def is_empty(self):
        return len(self.items) == 0
            
    def push(self, item):
        if type(item) != self.stack_type:
            raise RuntimeError("You can only create a Stack for one type, either Matrix or Stack.")
            
        self.items.append(item)
        
    def pop(self, index = -1):
        if not self.is_empty():
            return self.items.pop(index)
        else:
            raise IndexError("Cannot pop from an empty stack.")
        
    def clear(self):
        if not self.is_empty():
            self.items.clear()
        
    def __getitem__(self, index):
        return self.items[index]
    
    def __len__(self):
        return len(self.items)
    
    # TODO: Implement 'shape' property.