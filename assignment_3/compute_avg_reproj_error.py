import numpy as np 

def compute_avg_reproj_error(_M, _F):
    N = _M.shape[0]

    X = np.c_[ _M[:,0:2] , np.ones( (N,1) ) ].transpose()
    L = np.matmul( _F , X ).transpose()
    norms = np.sqrt( L[:,0]**2 + L[:,1]**2 )
    L = np.divide( L , np.kron( np.ones( (3,1) ) , norms ).transpose() )
    L = ( np.multiply( L , np.c_[ _M[:,2:4] , np.ones( (N,1) ) ] ) ).sum(axis=1)
    error = (np.fabs(L)).sum()

    X = np.c_[_M[:, 2:4], np.ones((N, 1))].transpose()
    L = np.matmul(_F.transpose(), X).transpose()
    norms = np.sqrt(L[:, 0] ** 2 + L[:, 1] ** 2)
    L = np.divide(L, np.kron(np.ones((3, 1)), norms).transpose())
    L = ( np.multiply( L , np.c_[ _M[:,0:2] , np.ones( (N,1) ) ] ) ).sum(axis=1)
    error += (np.fabs(L)).sum()

    return error/(N*2)