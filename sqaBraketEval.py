#    file:  sqaTerm.py
#  author:  Eric Neuscamman
#    date:  March 30, 2009
# summary:  Defines the term class and some functions that use it.
#
# (c) 2008-2009 Eric Neuscamman (eric.neuscamman@gmail.com)
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# In addition, any modification or use of this software should
# cite the following paper:
#
#   E. Neuscamman, T. Yanai, and G. K.-L. Chan.
#   J. Chem. Phys. 130, 124102 (2009)


from sqaTerm import term, termChop, combineTerms
from sqaTensor import kroneckerDelta, creOp, desOp
from sqaMisc import makeTuples, allDifferent
from sqaNormalOrder import normalOrder
from sqaOptions import options
import time

# The term class represents a set of constants and tensors that have been multiplied together.
# The class consists of a numerical constant factor, a list of named constants, and a list
# of tensors.
#
# It is common to encounter a list of term objects, which is often used to represent an expression.
# For example, the Hamiltonian in second quantization can be written as a list of terms.


#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------


def braketEval (terms):
  """
  terms must be a list of terms in normal order.
  Evaluates each term with the ground state closed-shell wave function <gs|terms|gs>
  It removes terms that delete electrons from unoccupied MOs but it makes nothing to cre/des
  ops acting on the active space.
  terms is modified on exit
  """

  if type(terms) != type([]) or not isinstance(terms[0], term):
    raise TypeError, "inList must be a list of sqa terms"

  startTime = time.time()

  if options.verbose:
    print("")
    print("Evaluate <gs|terms|gs>")

  # Iterate over terms in the list
  i = 0
  for t in terms:

    # If the number of cre/des operators is not the same, the term vanishes
    if t.nCreOps() != t.nDesOps():
      terms[i].numConstant = 0.0

    # Because we use the ground state as bra and ket, 
    # the terms with cre(virtual) or des(virtual) vanish
    for op in t.tensors:
      if isinstance(op, creOp) or isinstance(op, desOp):
        if (options.virtual_type in op.indices[0].indType) and \
           ((not options.core_type in op.indices[0].indType) and \
           (not options.active_type in op.indices[0].indType)):
          terms[i].numConstant = 0.0
          continue
 
    i += 1

  # Remove terms with 0 coefficient
  termChop(terms)

  # Move annihilation operators to the left
  # Iterate over terms in the list
  i = 0
  outTerms = []
  for t in terms:

    # Separate cre/des operators from other tensors
    ops = []
    nonOps = []
    for op in t.tensors:
      if isinstance(op, creOp) or isinstance(op, desOp):
        ops.append(op.copy())
      else:
        nonOps.append(op.copy())

    # Generate all contraction pairs
    contractionPairs = []
    for iop in range(1,len(ops)):
      rop = ops[iop]
      # Skip creation operators and operators with dummy indices
      if isinstance(rop, creOp):
        continue
      for jop in range(iop-1,-1,-1):
        lop = ops[jop]
        # Ignore destruction operators with active or closed-shell indices
        if isinstance(lop, creOp):
          contractionPairs.append((jop,iop));

    # Determine maximum contraction order
    creCount = 0
    maxConOrder = 0
    for iop in range(len(ops)):
      rop = ops[iop]
      move = isinstance(rop,desOp)
      if not move:
        creCount +=1
      elif move and creCount > 0:
        maxConOrder += 1
        creCount -= 1
    del(creCount)

    # Generate all contractions
    contractions = []
    for iord in range(maxConOrder+1):
      subCons = makeTuples(iord,contractionPairs)
      jord = 0
      while jord < len(subCons):
        creOpTags = []
        desOpTags = []
        for k in range(iord):
          creOpTags.append(subCons[jord][k][1])
          desOpTags.append(subCons[jord][k][0])
        if allDifferent(creOpTags) and allDifferent(desOpTags):
          jord += 1
        else:
          del(subCons[jord])
      for jord in range(len(subCons)):
        contractions.append(subCons[jord])
    del(subCons,creOpTags,desOpTags,contractionPairs)

    # For each contraction, generate the resulting term
    for contraction in contractions:
      conSign = 1
      deltaFuncs = []
      conIndeces = []
      subOpString = []
      subOpString.extend(ops)
      for conPair in contraction:
        index1 = ops[conPair[0]].indices[0]
        index2 = ops[conPair[1]].indices[0]
        deltaFuncs.append(kroneckerDelta([index1,index2]))
        subOpString[conPair[0]] = 'contracted'
        subOpString[conPair[1]] = 'contracted'
        for q in subOpString[conPair[0]+1:conPair[1]]:
          if not (q is 'contracted'):
            conSign *= -1
      iterm = 0
      while iterm < len(subOpString):
        if subOpString[iterm] is 'contracted':
          del(subOpString[iterm])
        else:
          iterm += 1
      # Generate permutations with permutation sign of all the terms
      # according to the contraction
      # In which position is the operator to be moved?
      #(sortSign,sortedOps) = sortOps(subOpString)
      sortedOps = subOpString + []
      i = 0
      sortSign = 1
      # Loop by pairs of operators, (left operator ,right operator)
      while i < len(sortedOps)-1:
        # Do we need to move the right operator to the left?
        move_rop_left = isinstance(sortedOps[i+1],desOp)
        # Can we move the left operator to the right?
        move_lop_right = not (isinstance(sortedOps[i],desOp))
        # Swap them if needed and update permutation sign
        if move_lop_right and move_rop_left:
          temp = sortedOps[i]
          sortedOps[i] = sortedOps[i+1]
          sortedOps[i+1] = temp
          i = 0
          sortSign *= -1
        else:
          i += 1

      totalSign = conSign * sortSign
      outTensors = []
      outTensors.extend(nonOps)
      outTensors.extend(deltaFuncs)
      outTensors.extend(sortedOps)
      outTerms.append( term(totalSign * t.numConstant, t.constants, outTensors) )

  # Remove terms with annihilation operators acting on occupied MOs on the bra
  # Iterate over terms in the list
  i = 0
  for t in outTerms:

    # Because we use the ground state as bra and ket, 
    # the terms with des(occ) on the left vanish 
    for op in t.tensors:
      # Check only the left most operator
      if isinstance(op, creOp) or isinstance(op, desOp):
        if isinstance(op, desOp):
          outTerms[i].numConstant = 0.0
        continue
 
    i += 1

  # Colapsing delta functions
  for t in outTerms:
    t.contractDeltaFuncs()
  termChop(outTerms)

  # Merge terms
  output = []
  for t in outTerms:
    output.extend(normalOrder(t))
  for t in output:
    t.makeCanonical()
  save = options.verbose
  options.verbose = False
  combineTerms(output)
  options.verbose = save

  # Print output
  if options.verbose:
    for t in output:
      print t

    print "Finished in %.3f seconds" %(time.time() - startTime)
    print ""

  return(output)

#--------------------------------------------------------------------------------------------------
