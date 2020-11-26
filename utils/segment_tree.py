import numpy as np

def is_tuple(val):
	return type(val) in [list,tuple]

class SegmentTree(object):
	__slots__ = ('_capacity','_value')
	
	def __init__(self, capacity, neutral_element):
		"""Build a Segment Tree data structure.
		https://en.wikipedia.org/wiki/Segment_tree
		Can be used as regular array, but with two
		important differences:
			a) setting item's value is slightly slower.
			   It is O(log capacity) instead of O(1).
			b) user has access to an efficient ( O(log segment size) )
			   `reduce` operation which reduces `operation` over
			   a contiguous subsequence of items in the array.
		Paramters
		---------
		capacity: int
			Total size of the array - must be a power of two.
		operation: lambda obj, obj -> obj
			and operation for combining elements (eg. sum, max)
			must form a mathematical group together with the set of
			possible values for array elements (i.e. be associative)
		neutral_element: obj
			neutral element for the operation above. eg. float('-inf')
			for max and 0 for sum.
		"""
		assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
		self._capacity = capacity
		self._value = [neutral_element]*(2 * capacity)
		self._neutral_element = neutral_element
		self._inserted_elements = 0

	def _reduce_helper(self, start, end, node, node_start, node_end): # O(log)
		if (start == node_start and end == node_end) or node_start >= node_end:
			return self._value[node]
		mid = (node_start + node_end) // 2
		if end <= mid:
			return self._reduce_helper(start, end, 2 * node, node_start, mid)
		else:
			if mid + 1 <= start:
				return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
			else:  
				return self._operation(
					self._reduce_helper(start, mid, 2 * node, node_start, mid),
					self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
				)

	def reduce(self, start=0, end=None): # O(log)
		"""Returns result of applying `self.operation`
		to a contiguous subsequence of the array.
			self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))
		Parameters
		----------
		start: int
			beginning of the subsequence
		end: int
			end of the subsequences
		Returns
		-------
		reduced: obj
			result of reducing self.operation over the specified range of array elements.
		"""
		if end is None:
			end = self._capacity
		if end < 0:
			end += self._capacity
		end -= 1
		return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

	def __setitem__(self, idx, val): # O(log)
		# index of the leaf
		idx += self._capacity
		if self._value[idx] == self._neutral_element:
			if val:
				self._inserted_elements += 1
		else:
			if not val:
				self._inserted_elements -= 1
		self._value[idx] = val if val else self._neutral_element
		idx //= 2
		while idx >= 1:
			self._value[idx] = self._operation(
				self._value[2 * idx],
				self._value[2 * idx + 1]
			)
			idx //= 2

	def __getitem__(self, idx): # O(1)
		assert 0 <= idx < self._capacity
		return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
	def __init__(self, capacity, neutral_element=0.):
		super(SumSegmentTree, self).__init__(
			capacity=capacity,
			neutral_element=neutral_element
		)
		self.min_tree = MinSegmentTree(capacity)
	
	@staticmethod
	def _operation(a, b):
		return a+b

	def __setitem__(self, idx, val): # O(log)
		super().__setitem__(idx, val)
		self.min_tree[idx] = val

	def sum(self, start=0, end=None, scaled=True): # O(log)
		"""Returns arr[start] + ... + arr[end]"""
		tot = super(SumSegmentTree, self).reduce(start, end)
		if scaled:
			tot -= self.min_tree.min()*self._inserted_elements
		return tot

	def find_prefixsum_idx(self, prefixsum_fn, scaled_prefix=True): # O(log)
		"""Find the highest index `i` in the array such that
			sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
		if array values are probabilities, this function
		allows to sample indexes according to the discrete
		probability efficiently.
		Parameters
		----------
		perfixsum: float
			upperbound on the sum of array prefix
		Returns
		-------
		idx: int
			highest index satisfying the prefixsum constraint
		"""
		# assert 0 <= prefixsum <= self.sum() + 1e-5 # O(1)
		mass = super(SumSegmentTree, self).reduce(start=0, end=None) # O(1)
		if scaled_prefix: # Use it in case of negative elements in the sumtree, they would break the tree invariant
			minimum = min(self._neutral_element,self.min_tree.min()) # O(1)
			summed_elements = self._capacity
			mass -= minimum*summed_elements
		prefixsum = prefixsum_fn(mass)
		idx = 1
		while idx < self._capacity:  # while non-leaf
			idx *= 2
			summed_elements /= 2
			value = self._value[idx]
			if scaled_prefix:
				value -= minimum*summed_elements
			if value <= prefixsum:
				prefixsum -= value
				idx += 1
		return idx - self._capacity
	
class MinSegmentTree(SegmentTree):
	def __init__(self, capacity, neutral_element=float('inf')):
		super(MinSegmentTree, self).__init__(
			capacity=capacity,
			neutral_element=neutral_element
		)
		
	@staticmethod
	def _operation(a, b):
		return a if a < b else b

	def min(self, start=0, end=None): # O(log)
		"""Returns min(arr[start], ...,  arr[end])"""
		return super(MinSegmentTree, self).reduce(start, end)

class MaxSegmentTree(SegmentTree):
	def __init__(self, capacity, neutral_element=float('-inf')):
		super(MinSegmentTree, self).__init__(
			capacity=capacity,
			neutral_element=neutral_element
		)
		
	@staticmethod
	def _operation(a, b):
		return a if a > b else b

	def max(self, start=0, end=None): # O(log)
		"""Returns min(arr[start], ...,  arr[end])"""
		return super(MaxSegmentTree, self).reduce(start, end)

# test = SumSegmentTree(4)
# test[2] = -10
# test[3] = -5
# test[0] = 1
# test[1] = 2
# print(test.sum())
# i = test.find_prefixsum_idx(23)
# print(i,test[i] )
