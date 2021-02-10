import math


def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def get_angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))