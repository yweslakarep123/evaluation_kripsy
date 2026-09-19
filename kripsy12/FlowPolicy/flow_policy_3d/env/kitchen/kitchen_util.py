import struct
import numpy as np

def parse_mjl_logs(read_filename, skipamount):
    with open(read_filename, mode='rb') as file:
        fileContent = file.read()
    if len(fileContent) < 28:
        raise ValueError(f"MJL terlalu pendek (header): {read_filename}")

    headers = struct.unpack('iiiiiii', fileContent[:28])
    nq = headers[0]
    nv = headers[1]
    nu = headers[2]
    nmocap = headers[3]
    nsensordata = headers[4]
    nuserdata = headers[5]
    name_len = headers[6]
    if name_len < 0 or 28 + name_len > len(fileContent):
        raise ValueError(f"MJL header name_len tidak valid: {read_filename}")

    name = struct.unpack(str(name_len) + 's', fileContent[28:28+name_len])[0]
    payload = fileContent[28 + name_len:]
    # Buang sisa byte yang bukan kelipatan 4 (file terpotong).
    leftover_bytes = len(payload) % 4
    if leftover_bytes:
        payload = payload[: len(payload) - leftover_bytes]
    num_floats = len(payload) // 4
    recsz = 1 + nq + nv + nu + 7 * nmocap + nsensordata + nuserdata
    if recsz <= 1:
        raise ValueError(f"MJL recsz tidak valid ({recsz}): {read_filename}")
    if num_floats < recsz:
        raise ValueError(
            f"MJL tidak punya satu frame lengkap: {read_filename}"
        )

    dat = np.asarray(struct.unpack(str(num_floats) + 'f', payload))
    leftover_floats = num_floats % recsz
    if leftover_floats:
        # Demo Kitchen resmi: beberapa file (sering indeks glob 481/482)
        # terpotong di frame terakhir. Pakai frame lengkap saja.
        n_records = num_floats // recsz
        print(
            f"[warn] MJL terpotong, memotong {leftover_floats} float sisa "
            f"({n_records} frame dipakai): {read_filename}"
        )
        dat = dat[: n_records * recsz]
    dat = np.reshape(dat, (-1, recsz)).T

    n_steps = int(dat.shape[1])
    skip = int(skipamount) if skipamount is not None else 1
    if skip < 1:
        skip = 1
    if n_steps < skip:
        skip = 1

    time = dat[0, :][::skip] - 0 * dat[0, 0]
    qpos = dat[1:nq + 1, :].T[::skip, :]
    qvel = dat[nq+1:nq+nv+1,:].T[::skip, :]
    ctrl = dat[nq+nv+1:nq+nv+nu+1,:].T[::skip,:]
    mocap_pos = dat[nq+nv+nu+1:nq+nv+nu+3*nmocap+1,:].T[::skip, :]
    mocap_quat = dat[nq+nv+nu+3*nmocap+1:nq+nv+nu+7*nmocap+1,:].T[::skip, :]
    sensordata = dat[nq+nv+nu+7*nmocap+1:nq+nv+nu+7*nmocap+nsensordata+1,:].T[::skip,:]
    userdata = dat[nq+nv+nu+7*nmocap+nsensordata+1:,:].T[::skip,:]

    if qpos.shape[0] == 0 or ctrl.shape[0] == 0:
        raise ValueError(f"MJL kosong setelah subsample: {read_filename}")

    data = dict(nq=nq,
               nv=nv,
               nu=nu,
               nmocap=nmocap,
               nsensordata=nsensordata,
               name=name,
               time=time,
               qpos=qpos,
               qvel=qvel,
               ctrl=ctrl,
               mocap_pos=mocap_pos,
               mocap_quat=mocap_quat,
               sensordata=sensordata,
               userdata=userdata,
               logName = read_filename
               )
    return data
