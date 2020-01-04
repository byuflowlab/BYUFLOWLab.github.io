subroutine laplacegridfortran(n, top, bottom, left, right, tol, iter_max, &
    phi, err_max, iters)

    implicit none
    integer, parameter :: dp = kind(0.d0)

    ! inputs
    real(dp), intent(in) :: top, bottom, left, right, tol
    integer, intent(in) :: n, iter_max

    ! outputs
    real(dp), intent(out), dimension(n+1, n+1) :: phi
    real(dp), intent(out) :: err_max
    integer, intent(out) :: iters

    ! local
    integer :: i, j
    real(dp) :: phi_prev


    phi(1, :) = bottom
    phi(n+1, :) = top
    phi(:, 1) = left
    phi(:, n+1) = right

    iters = 0
    err_max = 1e6

    do while (err_max > tol .and. iters < iter_max)

        err_max = -1.0

        ! iterating over interior points
        do i = 2, n
            do j = 2, n

                ! save previous point
                phi_prev = phi(i, j)

                ! update point
                phi(i, j) = (phi(i-1,j) + phi(i+1,j) + phi(i,j-1) + phi(i,j+1))/4.0

                ! update maximum error
                err_max = max(err_max, abs(phi(i, j) - phi_prev))
            end do
        end do

        iters = iters + 1

    end do

end subroutine laplacegridfortran
