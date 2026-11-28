"""Fee management routes.

Students cannot mark a fee as paid. They submit a claim; an admin checks it against
the college's own records and approves or rejects it. There is no payment gateway.
"""
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models import FeeStructure, Payment, PaymentClaim, StudentFee
from backend.core.security import get_current_student, require_role

router = APIRouter(prefix="/fees", tags=["Fees"])

@router.get("/my-fees")
def my_fees(student=Depends(get_current_student), db: Session = Depends(get_db)):
    fees = db.query(StudentFee).filter(StudentFee.student_id == student.id).all()
    result = []
    for f in fees:
        structure = db.query(FeeStructure).filter(FeeStructure.id == f.fee_structure_id).first()
        result.append({"id": f.id, "name": structure.name if structure else "Fee", "due": f.amount_due, "paid": f.amount_paid, "remaining": f.amount_due - f.amount_paid, "status": f.status, "due_date": f.due_date})
    return {"fees": result}

@router.get("/summary")
def fee_summary(student=Depends(get_current_student), db: Session = Depends(get_db)):
    fees = db.query(StudentFee).filter(StudentFee.student_id == student.id).all()
    total_due = sum(f.amount_due for f in fees)
    total_paid = sum(f.amount_paid for f in fees)
    overdue = sum(1 for f in fees if f.status == "Overdue")
    return {"total_due": total_due, "total_paid": total_paid, "pending": total_due - total_paid, "overdue_count": overdue}

class PaymentReq(BaseModel):
    student_fee_id: int
    amount: float = Field(gt=0, allow_inf_nan=False)
    payment_method: str = Field(default="upi", max_length=40)
    transaction_id: Optional[str] = Field(default=None, max_length=100)


def _claim_json(claim: PaymentClaim) -> dict:
    return {
        "id": claim.id,
        "student_fee_id": claim.student_fee_id,
        "amount": claim.amount,
        "method": claim.payment_method,
        "txn_id": claim.transaction_id,
        "status": claim.status,
        "submitted_at": str(claim.created_at),
    }


@router.post("/pay", status_code=202)
def submit_payment_claim(data: PaymentReq, student=Depends(get_current_student), db: Session = Depends(get_db)):
    fee = db.query(StudentFee).filter(StudentFee.id == data.student_fee_id, StudentFee.student_id == student.id).first()
    if not fee:
        raise HTTPException(404, "Fee not found")
    pending = db.query(PaymentClaim).filter(PaymentClaim.student_fee_id == fee.id, PaymentClaim.status == "Pending").all()
    open_amount = fee.amount_due - fee.amount_paid - sum(c.amount for c in pending)
    if data.amount > open_amount + 0.005:
        raise HTTPException(400, "Amount is more than the unpaid balance not already awaiting review")
    claim = PaymentClaim(
        institution_id=fee.institution_id,
        student_fee_id=fee.id,
        student_id=student.id,
        amount=data.amount,
        payment_method=data.payment_method,
        transaction_id=data.transaction_id,
    )
    db.add(claim)
    db.commit()
    return {"message": "Payment submitted for verification", "claim_id": claim.id, "status": claim.status}


@router.get("/claims")
def my_claims(student=Depends(get_current_student), db: Session = Depends(get_db)):
    claims = db.query(PaymentClaim).filter(PaymentClaim.student_id == student.id).order_by(PaymentClaim.id.desc()).all()
    return {"claims": [_claim_json(c) for c in claims]}


@router.get("/admin/claims")
def claims_for_review(status: str = "Pending", _=Depends(require_role("admin")), db: Session = Depends(get_db)):
    claims = db.query(PaymentClaim).filter(PaymentClaim.status == status).order_by(PaymentClaim.id).all()
    return {"claims": [{**_claim_json(c), "student_id": c.student_id} for c in claims]}


def _close_claim(db: Session, claim_id: int, new_status: str, admin_id: int) -> PaymentClaim:
    """Move a claim out of Pending. The conditional UPDATE stops two reviews both succeeding."""
    changed = (
        db.query(PaymentClaim)
        .filter(PaymentClaim.id == claim_id, PaymentClaim.status == "Pending")
        .update({"status": new_status, "reviewed_by": admin_id, "reviewed_at": datetime.now(timezone.utc)})
    )
    claim = db.query(PaymentClaim).filter(PaymentClaim.id == claim_id).first()
    if not claim:
        raise HTTPException(404, "Claim not found")
    if not changed:
        raise HTTPException(409, "Claim has already been reviewed")
    return claim


@router.post("/admin/claims/{claim_id}/approve")
def approve_claim(claim_id: int, admin=Depends(require_role("admin")), db: Session = Depends(get_db)):
    claim = _close_claim(db, claim_id, "Approved", admin["entity_id"])
    fee = db.query(StudentFee).filter(StudentFee.id == claim.student_fee_id).with_for_update().first()
    if claim.amount > fee.amount_due - fee.amount_paid + 0.005:
        db.rollback()
        raise HTTPException(409, "Claim is more than the unpaid balance")
    payment = Payment(
        institution_id=claim.institution_id,
        student_fee_id=fee.id,
        student_id=claim.student_id,
        amount=claim.amount,
        payment_method=claim.payment_method,
        transaction_id=claim.transaction_id,
    )
    db.add(payment)
    db.flush()
    claim.payment_id = payment.id
    fee.amount_paid += claim.amount
    fee.status = "Paid" if fee.amount_paid >= fee.amount_due - 0.005 else "Partial"
    db.commit()
    return {"message": "Payment recorded", "payment_id": payment.id, "fee_status": fee.status}


@router.post("/admin/claims/{claim_id}/reject")
def reject_claim(claim_id: int, admin=Depends(require_role("admin")), db: Session = Depends(get_db)):
    _close_claim(db, claim_id, "Rejected", admin["entity_id"])
    db.commit()
    return {"message": "Claim rejected"}


@router.get("/payments")
def payment_history(student=Depends(get_current_student), db: Session = Depends(get_db)):
    payments = db.query(Payment).filter(Payment.student_id == student.id).order_by(Payment.paid_at.desc()).all()
    return {"payments": [{"id": p.id, "amount": p.amount, "method": p.payment_method, "txn_id": p.transaction_id, "date": str(p.paid_at)} for p in payments]}
