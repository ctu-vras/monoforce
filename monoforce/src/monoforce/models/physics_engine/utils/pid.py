from torch import Tensor


class PID:
    def __init__(self, kp, ki, kd, max_output, min_output):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max = max_output
        self.min = min_output
        self.i = 0.0
        self.e_prev = 0.0

    def reset(self, i: float | Tensor = 0.0, e_prev: float | Tensor = 0.0):
        self.i = 0.0
        self.e_prev = 0.0

    def set_gains(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def step(self, setpoint, process_variable, dt):
        """
        PID controller step

        Args:
          setpoint - desired value
          process_variable - current value
          dt - time step

        Returns:
            output - PID output
        """

        # Regulation step
        error = setpoint - process_variable
        self.i += error * dt
        output = self.kp * error + self.ki * self.i + self.kd * (error - self.e_prev) / dt
        self.e_prev = error

        # Anti-windup (conditional integration) and saturation
        max_mask = output > self.max
        output[max_mask] = self.max
        self.i[max_mask] -= error[max_mask] * dt
        min_mask = output < self.min
        output[min_mask] = self.min
        self.i[min_mask] -= error[min_mask] * dt

        return output
